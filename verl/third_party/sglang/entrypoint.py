import asyncio
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union
import uuid
from sglang.srt.managers.io_struct import GenerateReqInput, AbortReq
from sglang.srt.entrypoints.verl_engine import VerlEngine as VerlEngineBase
from sglang.srt.entrypoints.verl_engine import _preprocess_tensor_for_update_weights
from sglang.srt.server import Engine
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj
from sglang.srt.model_executor.model_runner import LocalSerializedTensor

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor

# Helper function to get first value from an async generator
async def get_first_value(async_gen):
    return await async_gen.__anext__()

class CustomEngine(Engine):

    def gather_partial_outputs(self):
        partial_outputs = {}
        for _, recv_obj in self.tokenizer_manager.orphan_outputs.items():
            for i, rid in enumerate(recv_obj.rids):
                if recv_obj.finished_reasons[i] and recv_obj.finished_reasons[i]['type'] == 'abort':
                    output_strs = recv_obj.output_strs[i]
                    partial_outputs[rid] = output_strs

        partial_output_ids = {}
        for rid, output_strs in partial_outputs.items():
            partial_output_ids[rid] = self.tokenizer_manager.tokenizer.encode(output_strs)
        outputs = []
        for rid, output_ids in partial_output_ids.items():
            meta_info = {
                "id": rid,
                "finish_reason": {'type': 'abort'},
            }
            output_dict  = {
                "output_ids": output_ids,
                "meta_info": meta_info,
            }
            outputs.append(output_dict)
        self.tokenizer_manager.orphan_outputs = {}
        return outputs

    async def get_first_n_results(self, tasks, num_returns):
        outputs = {}
        completed_oids = []
        completed_rids = []
        all_tasks = [task for task_dict in tasks.values() for task in task_dict.values()]
        for oid in tasks.keys():
            outputs[oid] = {}
        for task in asyncio.as_completed(all_tasks):
            result = await task
            rid = result['meta_info']['id']
            oid = rid.split('_nid')[0]
            outputs[oid][rid] = result
            completed_rids.append(rid)
            tasks[oid].pop(rid)
            if len(tasks[oid].keys()) == 0:
                completed_oids.append(oid)
                tasks.pop(oid)
            if len(completed_oids) >= num_returns:
                break

        for oid, task_dict in tasks.items():
            for rid in task_dict.keys():
                self.tokenizer_manager.abort_request(rid)

        # Wait for idle
        while True:
            # print(f'waiting for idle')
            internal_state = await self.tokenizer_manager.get_internal_state()
            if internal_state['is_idle']:
                # print(f'idle')
                break
        
        # Scavenge incomplete results
        incomplete_tasks = [task for task_dict in tasks.values() for task in task_dict.values()]

        await asyncio.sleep(1)
        for task in incomplete_tasks:
            if task.done():
                result = await task
                incomplete_tasks.remove(task)
                rid = result['meta_info']['id']
                oid = rid.split('_nid')[0]
                outputs[oid][rid] = result
                finish_reason = result['meta_info'].get('finish_reason', {type: ''})
                # to_delete_rids.remove(rid)
                if finish_reason['type'] != 'abort':
                    completed_rids.append(rid)
                    tasks[oid].pop(rid)
                    if len(tasks[oid].keys()) == 0:
                        completed_oids.append(oid)
                        tasks.pop(oid)
            else:
                task.cancel()

        partial_outputs = self.gather_partial_outputs()

        self.tokenizer_manager.clear_queue()
        for output in partial_outputs:
            rid = output['meta_info']['id']
            oid = rid.split('_nid')[0]
            outputs[oid][rid] = output

        return outputs

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
        stream: bool = False,
        rid: Optional[Union[List[str], str]] = None,
        num_returns: Optional[int] = None,
    ) -> Union[Dict]:
        if prompt is not None:
            batch_size = len(prompt)
        if input_ids is not None:
            batch_size = len(input_ids)
        loop = asyncio.get_event_loop()
        
        assert num_returns is not None, "num_returns should be provided"
        assert rid is not None, "rid should be provided"

        original_rids = list(set([rid.split('_nid')[0] for rid in rid]))
        all_rids = rid


        objs = {}
        tasks = {}
        for oid in original_rids:
            objs[oid] = {}
            tasks[oid] = {}
        for i, rid in enumerate(all_rids):
            oid = rid.split('_nid')[0]
            # tmp_sampling_params = sampling_params_.copy()
            # max_new_tokens = sampling_params_.get('max_new_tokens', 16*1024)
            # max_length = 18 * 1024
            # context_length = len(input_ids[i])
            # max_new_tokens = min(max_new_tokens, max_length - context_length)
            # max_new_tokens = max(max_new_tokens, 1)
            # tmp_sampling_params['max_new_tokens'] = max_new_tokens
            objs[oid][rid] = GenerateReqInput(
                text=None,
                input_ids=input_ids[i],
                sampling_params=sampling_params[i],
                image_data=None,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=top_logprobs_num,
                token_ids_logprob=token_ids_logprob,
                lora_path=lora_path,
                custom_logit_processor=custom_logit_processor,
                stream=False,
                rid=rid,
            )
            generator = self.tokenizer_manager.generate_request(objs[oid][rid], None)
            task = loop.create_task(get_first_value(generator))
            tasks[oid][rid] = task



        outputs = loop.run_until_complete(self.get_first_n_results(tasks, num_returns))

        return outputs

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
        return None

    def resume_memory_occupation(self):
        if self._tp_rank == 0:
            self._engine.resume_memory_occupation()
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
            else:
                output = self._engine.custom_generate(
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
        else:
            output = None

        # # Most naive implementation, can extract tensor and send via gloo if too slow
        [output] = broadcast_pyobj(
            data=[output],
            rank=self._tp_rank,
            dist_group=self._device_mesh_cpu.get_group(),
            src=self._device_mesh_cpu.mesh[0].item(),
        )
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

        return output