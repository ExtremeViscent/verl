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

class Entrypoint(SpmdEntrypoint):

    def generate(self, obj: GenerateReqInput, num_return_sequences: Optional[int] = None):
        obj.normalize_batch_and_arguments()
        objs = [obj] if obj.is_single else [obj[i] for i in range(obj.batch_size)]
        tokenized_requests = self._generation_converter.tokenize_requests(objs)
        rid_to_req_index = {r.rid: i for i, r in enumerate(tokenized_requests)}

        outputs: List[Dict[str, Any]] = [None] * obj.batch_size

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
        pending_rids = []
        completed_rids = []
        while self._scheduler.process_batch():
            if num_return_sequences is not None:
                ret_count = 0
                finished_outputs = []
                pending_rids = [r.rid for r in objs]
                for output in outputs:
                    if output is not None and output.get("meta_info", {}).get("finish_reason", None) is not None:
                        ret_count += 1
                        pending_rids.remove(output["id"])
                        completed_rids.append(output["id"])
                    if ret_count >= num_return_sequences:
                        for rid in pending_rids:
                            self._scheduler.abort_request(AbortReq(rid=rid))
                        for output in outputs:
                            if output is not None:
                                finished_outputs.append(output)
                        break
            else:
                pass

        return finished_outputs, completed_rids, pending_rids

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
        return self._generate_impl(obj, num_return_sequences)

    def _generate_impl(self, obj: GenerateReqInput, num_return_sequences: Optional[int] = None):
        return self._entrypoint.generate(obj, num_return_sequences=num_return_sequences)

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
