import asyncio
from typing import AsyncIterator, Dict, List, Optional, Union
import uuid
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.entrypoints.verl_engine import VerlEngine as VerlEngineBase
from sglang.srt.server import Engine
from sglang.srt.utils import broadcast_pyobj

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
        stream: bool = False,
        rid: Optional[Union[List[str], str]] = None,
        num_returns: Optional[int] = None,
    ) -> Union[Dict]:
        # First try to get an existing event loop, create one only if necessary
        try:
            loop = asyncio.get_event_loop()
            created_loop = False
        except RuntimeError:
            # No running event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            created_loop = True
        
        try:
            batch_size = 0
            if prompt is not None:
                batch_size = len(prompt)
            if input_ids is not None:
                batch_size = len(input_ids)

            # Generate unique request IDs if not provided
            if rid is None:
                original_rids = [f"req_{i}_{uuid.uuid4().hex[:8]}" for i in range(batch_size)]
            else:
                original_rids = rid if isinstance(rid, list) else [rid]



            # Convert when n>1
            n = sampling_params.get("n", 1) if sampling_params is not None else 1
            sampling_params_ = sampling_params.copy()
            if n > 1:
                all_rids = []
                for i in range(batch_size):
                    all_rids.extend([original_rids[i]+f'_nid{uuid.uuid4().hex[:8]}' for j in range(n)])
                sampling_params_['n'] = 1
            else:
                all_rids = original_rids


            # Process each prompt individually to get results as they come in
            all_tasks = []
            rid_to_task = {}
            for i in range(batch_size):
                for j in range(n):
                    # Create a single-prompt request with a single rid
                    single_obj = GenerateReqInput(
                        text=prompt[i] if prompt is not None else None,
                        input_ids=[input_ids[i]] if input_ids is not None else None,
                        sampling_params=sampling_params_,
                        image_data=image_data[i] if image_data is not None else None,
                        return_logprob=return_logprob,
                        logprob_start_len=logprob_start_len,
                        top_logprobs_num=top_logprobs_num,
                        token_ids_logprob=token_ids_logprob,
                        lora_path=lora_path,
                        stream=stream,
                        custom_logit_processor=custom_logit_processor,
                        rid=[all_rids[i*n+j]],  # Pass as a list with single rid
                    )
                    # Create an async task for this prompt
                    generator = self.tokenizer_manager.generate_request(single_obj, None)
                    task = loop.create_task(get_first_value(generator))
                    all_tasks.append(task)
                    rid_to_task[all_rids[i*n+j]] = task

                
            # Wait for the first num_return_seqs tasks to complete
            num_returns = min(num_returns, batch_size)  # Don't try to get more results than prompts
            results, incomplete_rids = loop.run_until_complete(get_first_n_results(all_tasks, all_rids, num_returns, n))
            print(f'results: {len(results)}')
            print(f'num_returns: {num_returns}')
            print(f'n: {n}')
            print(f'incomplete_rids: {len(incomplete_rids)}')
            # Abort the incomplete requests
            for rid in incomplete_rids:
                self.tokenizer_manager.abort_request(rid)
                rid_to_task[rid].cancel()
            
            # Map incomplete rid to original rid
            completed_original_rids = []
            for i in range(num_returns):
                j = i*n
                completed_original_rids.append(results[j]['meta_info']['id'].split('_nid')[0])
            incomplete_original_rids = set(original_rids) - set(completed_original_rids)
            return results, completed_original_rids, incomplete_original_rids
        finally:
            # Only close the loop if we created it
            if created_loop:
                try:
                    loop.close()
                except:
                    pass

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
                output, completed_original_rids, incomplete_original_rids = self._engine.custom_generate(
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
            completed_original_rids = None
            incomplete_original_rids = None

        # Most naive implementation, can extract tensor and send via gloo if too slow
        [output] = broadcast_pyobj(
            data=[output],
            rank=self._tp_rank,
            dist_group=self._device_mesh_cpu.get_group(),
            src=self._device_mesh_cpu.mesh[0].item(),
        )
        if num_returns is not None:
            [incomplete_original_rids] = broadcast_pyobj(
                data=[incomplete_original_rids],
                rank=self._tp_rank,
                dist_group=self._device_mesh_cpu.get_group(),
                src=self._device_mesh_cpu.mesh[0].item(),
            )
            [completed_original_rids] = broadcast_pyobj(
                data=[completed_original_rids],
                rank=self._tp_rank,
                dist_group=self._device_mesh_cpu.get_group(),
                src=self._device_mesh_cpu.mesh[0].item(),
            )

        if num_returns is not None:
            return output, completed_original_rids, incomplete_original_rids
        else:
            return output