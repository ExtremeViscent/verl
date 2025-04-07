import asyncio
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union
import uuid
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.entrypoints.verl_engine import VerlEngine as VerlEngineBase
from sglang.srt.entrypoints.verl_engine import _preprocess_tensor_for_update_weights
from sglang.srt.server import Engine
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj
from sglang.srt.model_executor.model_runner import LocalSerializedTensor

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor


async def get_first_n_results(tasks, all_rids, num_returns, n=1):
    grouped = n > 1
    results = []
    incomplete_rids = set()
    completed_rids = set()
    if grouped:
        original_rids = [rid.split('_nid')[0] for rid in all_rids]
        original_rids = list(set(original_rids))
        n = len(all_rids) // len(original_rids)
        partial_results = {}
        for rid in original_rids:
            partial_results[rid] = []
        for task in asyncio.as_completed(tasks):
            result = await task
            result = result[0]
            original_rid = result['meta_info']['id'].split('_nid')[0]
            partial_results[original_rid].append(result)
            completed_rids.add(result['meta_info']['id'])
            if len(partial_results[original_rid]) == n:
                results.extend(partial_results[original_rid])
                del partial_results[original_rid]
                if len(results) == num_returns * n:
                    incomplete_rids = set(all_rids) - completed_rids
                    return results, incomplete_rids
    else:
        for task in asyncio.as_completed(tasks):
            result = await task
            result = result[0]
            results.append(result)
            completed_rids.add(result['meta_info']['id'])
            if len(results) == num_returns:
                incomplete_rids = set(all_rids) - completed_rids
                return results, incomplete_rids

# Helper function to get first value from an async generator
async def get_first_value(async_gen):
    return await async_gen.__anext__()

class CustomEngine(Engine):
    def custom_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[List[str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be a file name, a url, or base64 encoded string.
        # See also python/sglang/srt/utils.py:load_image.
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        stream: bool = True,
        rid: Optional[Union[List[str], str]] = None,
        num_returns: Optional[int] = None,
    ) -> Union[Dict]:
        batch_size = 0
        if prompt is not None:
            batch_size = len(prompt)
        if input_ids is not None:
            batch_size = len(input_ids)

        # # Generate unique request IDs if not provided
        # if rid is None:
        #     original_rids = [f"req_{i}_{uuid.uuid4().hex[:8]}" for i in range(batch_size)]
        # else:
        #     original_rids = rid if isinstance(rid, list) else [rid]

        # # Convert when n>1
        # n = sampling_params.get("n", 1) if sampling_params is not None else 1
        # sampling_params_ = sampling_params.copy()
        # if n > 1:
        #     all_rids = []
        #     for i in range(batch_size):
        #         all_rids.extend([original_rids[i]+f'_nid{uuid.uuid4().hex[:8]}' for j in range(n)])
        #     sampling_params_['n'] = 1
        # else:
        #     all_rids = [original_rids[i]+f'_nid{uuid.uuid4().hex[:8]}' for i in range(batch_size)]

        #Check prequsites
        n = sampling_params.get("n", 1) if sampling_params is not None else 1
        sampling_params_ = sampling_params.copy()
        sampling_params_['n'] = 1
        assert num_returns is not None, "num_returns should be provided"
        assert rid is not None, "rid should be provided"
        assert stream, "stream should be True"

        original_rids = list(set([rid.split('_nid')[0] for rid in rid]))
        all_rids = rid


        # extended_input_ids = []
        # for i in range(batch_size):
        #     for j in range(n):
        #         extended_input_ids.append(input_ids[i])

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params_,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            stream=stream,
            rid=all_rids,
        )
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)

        def generator_wrapper():
            while True:
                try:
                    chunk = loop.run_until_complete(generator.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break

        outputs = {}
        completed_rids = {}
        completed_oids = []

        for oid in original_rids:
            outputs[oid] = {}
            completed_rids[oid] = []

        wrapped_generator = generator_wrapper()

        cnt = 0
        for chunk in wrapped_generator:
            if chunk['meta_info']['finish_reason'] is not None:
                id = chunk['meta_info']['id']
                oid = id.split('_nid')[0]
                outputs[oid][id] = chunk
                if id not in completed_rids[oid]:
                    completed_rids[oid].append(id)
                    cnt += 1
                    if len(completed_rids[oid]) == n:
                        completed_oids.append(oid)
                        if len(completed_oids) >= num_returns:
                            break
        incomplete_oids = list(set(original_rids) - set(completed_oids))
        completed_rids = completed_rids.values()
        completed_rids = [item for sublist in completed_rids for item in sublist]
        incomplete_rids = list(set(all_rids) - set(completed_rids))

        for rid in incomplete_rids:
            self.tokenizer_manager.abort_request(rid)

        # self.tokenizer_manager.clear_queue()
        while True:
            print(f'waiting for idle')
            task = loop.create_task(self.tokenizer_manager.get_internal_state())
            internal_state = loop.run_until_complete(task)
            if internal_state['is_idle']:
                print(f'idle')
                break
            time.sleep(1)

        # incomplete_rids_ = incomplete_rids.copy()

        # for chunk in wrapped_generator:
        #     id = chunk['meta_info']['id']
        #     if id in incomplete_rids_:
        #         incomplete_rids_.remove(id)
        #         outputs[oid][id] = chunk
        #         print(f'{len(incomplete_rids_)}')
        #     if len(incomplete_rids_) == 0:
        #         break
                

        completed_outputs = {}
        incomplete_outputs = {}
        for oid in completed_oids:
            completed_outputs[oid] = outputs[oid]
        for oid in incomplete_oids:
            incomplete_outputs[oid] = outputs[oid]

        return completed_outputs, incomplete_outputs

class VerlEngine(VerlEngineBase):
    def __init__(
        self,
        device_mesh_cpu: DeviceMesh,
        nnodes: int = 1,
        **kwargs,
    ):
        self._device_mesh_cpu = device_mesh_cpu
        self._tp_rank = device_mesh_cpu.get_local_rank()
        self._tp_size = device_mesh_cpu.size()
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = CustomEngine(
                **kwargs, tp_size=self._tp_size, node_rank=node_rank, nnodes=nnodes
            )
        else:
            self._engine = None

        dist.barrier(group=self._device_mesh_cpu.get_group())

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
    ):
        # Most naive implementation, can optimize a lot if it is bottleneck
        for tensor_index, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(
                _preprocess_tensor_for_update_weights(tensor)
            )

            if self._tp_rank == 0:
                gathered_serialized_tensors = [None for _ in range(self._tp_size)]
            else:
                gathered_serialized_tensors = None
            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=self._device_mesh_cpu.mesh.tolist()[0],
                group=self._device_mesh_cpu.get_group(),
            )

            if self._tp_rank == 0:
                self._engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=tensor_index == len(named_tensors) - 1,
                )

    def release_memory_occupation(self):
        if self._tp_rank == 0:
            self._engine.release_memory_occupation()
        print(f"release_memory_occupation {self._tp_rank}")
        return None

    def resume_memory_occupation(self):
        if self._tp_rank == 0:
            self._engine.resume_memory_occupation()
        print(f"resume_memory_occupation {self._tp_rank}")
        return None

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be a file name, a url, or base64 encoded string.
        # See also python/sglang/srt/utils.py:load_image.
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        num_returns: Optional[int] = None,
        rid: Optional[Union[List[str], str]] = None,
    ):
        if self._tp_rank == 0:
            if num_returns is None:
                output = self._engine.generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    input_ids=input_ids,
                    image_data=image_data,
                    return_logprob=return_logprob,
                    logprob_start_len=logprob_start_len,
                    top_logprobs_num=top_logprobs_num,
                    token_ids_logprob=token_ids_logprob,
                    lora_path=lora_path,
                    custom_logit_processor=custom_logit_processor,
                )
                completed_original_rids = None
                incomplete_original_rids = None
            else:
                completed_outputs, incomplete_outputs = self._engine.custom_generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    input_ids=input_ids,
                    image_data=image_data,
                    return_logprob=return_logprob,
                    logprob_start_len=logprob_start_len,
                    top_logprobs_num=top_logprobs_num,
                    token_ids_logprob=token_ids_logprob,
                    lora_path=lora_path,
                    custom_logit_processor=custom_logit_processor,
                    num_returns=num_returns,
                    rid=rid,
                )
                output = completed_outputs
                completed_original_rids = None
                incomplete_original_rids = None

        # # Most naive implementation, can extract tensor and send via gloo if too slow
        # [output] = broadcast_pyobj(
        #     data=[output],
        #     rank=self._tp_rank,
        #     dist_group=self._device_mesh_cpu.get_group(),
        #     src=self._device_mesh_cpu.mesh[0].item(),
        # )
        # if num_returns is not None:
        #     [incomplete_original_rids] = broadcast_pyobj(
        #         data=[incomplete_original_rids],
        #         rank=self._tp_rank,
        #         dist_group=self._device_mesh_cpu.get_group(),
        #         src=self._device_mesh_cpu.mesh[0].item(),
        #     )
        #     [completed_original_rids] = broadcast_pyobj(
        #         data=[completed_original_rids],
        #         rank=self._tp_rank,
        #         dist_group=self._device_mesh_cpu.get_group(),
        #         src=self._device_mesh_cpu.mesh[0].item(),
        #     )

        if num_returns is not None:
            return completed_outputs, incomplete_outputs
        else:
            return output