import os
import sys
import datasets
sys.path.insert(0, "/home/aiscuser/verl")
from verl.utils.hdfs_io import copy, makedirs
import argparse

import re

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.workers.reward_manager import DAPORewardManager
from transformers import AutoTokenizer, AutoProcessor
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

tokenizer = AutoTokenizer.from_pretrained('/gpfs/models/huggingface.co/meta-llama/Meta-Llama-3___1-8B-Instruct')
processor = AutoProcessor.from_pretrained('/gpfs/models/huggingface.co/meta-llama/Meta-Llama-3___1-8B-Instruct')

def prepare_math500():
    data_source = 'HuggingFaceH4/MATH-500'

    dataset = datasets.load_dataset(data_source, 'default')

    dataset = dataset['test']

    instruction_following_1 = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n'
    instruction_following_2 = 'Remember to put your answer on its own line after "Answer:".'
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('problem')

            question = instruction_following_1 + '\n' + question_raw + '\n' + instruction_following_2

            answer_raw = example.pop('answer')
            data = {
                "data_source": 'math_500',
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "MATH",
                "reward_model": {
                    "style": "rule-lighteval/MATH_v2",
                    "ground_truth": answer_raw
                },
                "extra_info": {
                    'dummy': 'dummy',
                }
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn('test'), with_indices=True)
    return dataset

math500_dataset = prepare_math500()
math500_dataset.to_parquet('/gpfs/users/zhangyiqi/srl/data/math500_eval.parquet')