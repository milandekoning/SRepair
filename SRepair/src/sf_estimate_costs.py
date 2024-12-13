import re
import os
import time
import json
import openai
import argparse
import tiktoken
import numpy as np
from tqdm import tqdm
from gen_solution_prompt import sf_construct_prompt

class SolInfo():
    def __init__(self, dataset_path, sample_size, solution_path):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        self.sample_size = sample_size
        self.solution_path = solution_path
        return


def main():
    sol_info = SolInfo(
        args.d,
        args.s,
        args.o
    )

    with open(sol_info.solution_path, 'r') as f:
        raw_solutions = json.load(f)

    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-1106')
    tokens_per_prompt = []
    tokens_per_solution = []

    for bug_name in tqdm(sol_info.dataset.keys()):
        prompt = sf_construct_prompt(sol_info.dataset, bug_name)
        tokens = encoding.encode(prompt)
        tokens_per_prompt.append(len(tokens))



    for solution in raw_solutions['Chart-1']['solutions']:
        tokens = encoding.encode(solution)
        tokens_per_solution.append(len(tokens))

    cost_per_1M_input = 1.00
    cost_per_1M_output = 2.00
    mean_tokens_prompt = np.mean(tokens_per_prompt)
    mean_tokens_output = np.mean(tokens_per_solution)
    mean_input_cost = mean_tokens_prompt * cost_per_1M_input / 1000000
    mean_output_cost = mean_tokens_output * cost_per_1M_output / 1000000
    total_bug_cost = (mean_input_cost + mean_output_cost) * sol_info.sample_size
    total_cost = total_bug_cost * len(sol_info.dataset.keys())

    print(f"Mean tokens per bug prompt: {mean_tokens_prompt}")
    print(f"Mean tokens per bug output: {mean_tokens_output}")
    print(f"Mean cost per bug prompt: ${mean_input_cost}")
    print(f"Mean cost per bug output: ${mean_output_cost}")
    print(f"Total cost per bug: ${total_bug_cost}")
    print(f"Total cost: ${total_cost}")
    print(f"Total cost with batch API: ${total_cost / 2}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='dataset path')
    parser.add_argument('-s', type=int, required=False, help='sample_size', default=1)
    parser.add_argument('-o', type=str, required=True, help='raw_solution path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main()