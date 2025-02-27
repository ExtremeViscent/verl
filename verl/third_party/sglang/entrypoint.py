from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.distributed import ParallelProcessGroups
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.generation_manager import GenerationConverter
from sglang.srt.managers.io_struct import BatchTokenIDOut, GenerateReqInput
from sglang.srt.managers.scheduler import Scheduler, SchedulerCallback
from sglang.srt.server_args import ServerArgs
from sglang.srt.orchestration.spmd.entrypoint import Entrypoint as SpmdEntrypoint
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.server.engine_base import EngineBase
from uuid import uuid4

class Entrypoint(SpmdEntrypoint):

    def generate(self, 
                 obj: GenerateReqInput, 
                 num_return_sequences: Optional[int] = None, 
                 num_return_groups: Optional[int] = None):
        obj.normalize_batch_and_arguments()
        do_grpo = obj.parallel_sample_num > 1
        if do_grpo:
            original_rids = obj.rid
            objs = []
            for i in range(obj.batch_size):
                n = obj.parallel_sample_num
                for j in range(n):
                    new_obj = obj[i]
                    new_obj.rid = uuid4()
                    objs.append(new_obj)
        else:
            objs = [obj] if obj.is_single else [obj[i] for i in range(obj.batch_size)]
        tokenized_requests = self._generation_converter.tokenize_requests(objs)
        rid_to_req_index = {r.rid: i for i, r in enumerate(tokenized_requests)}

        outputs: List[Dict[str, Any]] = [None] * obj.batch_size * obj.parallel_sample_num

        def _handle_scheduler_output(batch_token_id_out: BatchTokenIDOut):
            batch_str_out = self._detokenizer.handle_batch_token_id_out(
                batch_token_id_out
            )
            for output_index in range(len(batch_str_out.rids)):
                req_index = rid_to_req_index[batch_str_out.rids[output_index]]
                outputs[req_index] = self._generation_converter.postprocess_response(
                    batch_str_out, index=output_index, req_obj=objs[req_index]
                )

        self._scheduler.callback = SchedulerCallback(
            on_generation_output=_handle_scheduler_output
        )

        for tokenized_request in tokenized_requests:
            self._scheduler.handle_generate_request(tokenized_request)

        finished_outputs = outputs
        while self._scheduler.process_batch():
            if num_return_groups is not None:
                finished_outputs = []
                completed_gids = []
                pending_gids = [i for i in range(obj.batch_size)]
                for i in range(obj.batch_size):
                    ret_count = 0
                    for j in range(obj.parallel_sample_num):
                        if outputs[i * obj.parallel_sample_num + j] is None or outputs[i * obj.parallel_sample_num + j].get("meta_info", {}).get("finish_reason", None) is None:
                            break
                        ret_count += 1
                    if ret_count >= obj.parallel_sample_num:
                        completed_gids.append(i)
                        pending_gids.remove(i)
                        if len(completed_gids) >= num_return_groups:
                            for gid in pending_gids:
                                for u in range(obj.parallel_sample_num):
                                    rid = objs[gid * obj.parallel_sample_num + u].rid
                                    self._scheduler.abort_request(AbortReq(rid=rid))
                            for gid in completed_gids:
                                for v in range(obj.parallel_sample_num):
                                    assert outputs[gid * obj.parallel_sample_num + v] is not None, f"idx: {gid * obj.parallel_sample_num + v} is None, completed_gids: {completed_gids}, pending_gids: {pending_gids}"
                                    assert outputs[gid * obj.parallel_sample_num + v].get("meta_info", {}).get("finish_reason", None) is not None, f"idx: {gid * obj.parallel_sample_num + v} finish_reason is None"
                                    finished_outputs.append(outputs[gid * obj.parallel_sample_num + v])
                            while self._scheduler.process_batch():
                                pass
                            completed_rids = [original_rids[i] for i in completed_gids]
                            pending_rids = [original_rids[i] for i in pending_gids]
                            return finished_outputs, completed_rids, pending_rids
            elif num_return_sequences is not None and num_return_sequences != obj.batch_size:
                ret_count = 0
                finished_outputs = []
                completed_rids = []
                pending_rids = [r.rid for r in objs]
                for output in outputs:
                    if output is not None and output.get("meta_info", {}).get("finish_reason", None) is not None:
                        ret_count += 1
                        pending_rids.remove(output["meta_info"]["id"])
                        completed_rids.append(output["meta_info"]["id"])
                    if ret_count >= num_return_sequences:
                        for rid in pending_rids:
                            self._scheduler.abort_request(AbortReq(rid=rid))
                        for output in outputs:
                            if output is not None and output["meta_info"]["id"] in completed_rids:
                                finished_outputs.append(output)
                        while self._scheduler.process_batch():
                            pass
                        return finished_outputs, completed_rids, pending_rids
            else:
                pass

        if num_return_sequences is not None:
            return finished_outputs, completed_rids, pending_rids
        else:
            return outputs

class EngineFragment(EngineBase):
    def __init__(
        self,
        nccl_port: int,
        gpu_id: int,
        tp_rank: int,
        parallel_process_groups: Optional[ParallelProcessGroups] = None,
        log_level: str = "error",
        *args,
        **kwargs,
    ):
        server_args = ServerArgs(*args, log_level=log_level, **kwargs)
        self._entrypoint = Entrypoint(
            server_args=server_args,
            nccl_port=nccl_port,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            parallel_process_groups=parallel_process_groups,
        )

    # Make it in base class to ensure API is exactly the same, as well as extracting common logic
    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        stream: bool = False,
        rid: Optional[Union[List[str], str]] = None,
        num_return_sequences: Optional[int] = None,
        num_return_groups: Optional[int] = None
    ):
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            lora_path=lora_path,
            stream=stream,
            rid=rid,
        )
        return self._generate_impl(obj, num_return_sequences, num_return_groups)

    def _generate_impl(self, obj: GenerateReqInput, num_return_sequences: Optional[int] = None, num_return_groups: Optional[int] = None):
        return self._entrypoint.generate(obj, num_return_sequences=num_return_sequences, num_return_groups=num_return_groups)

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
    ):
        self._entrypoint.update_weights_from_tensor(named_tensors, load_format)

    def release_gpu_occupation(self):
        self._entrypoint.release_gpu_occupation()

    def resume_gpu_occupation(self):
        self._entrypoint.resume_gpu_occupation()

    def shutdown(self):
        self._entrypoint.shutdown()
