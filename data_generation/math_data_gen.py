import json
import os
import re
import sys
from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = r"""
You are a helpful AI Assistant. First, think through the reasoning inside <think>...</think>.
Then, always present the final answer in \boxed{}.""".strip()

SYSTEM_PROMPT_MULTIPLE_CHOICE = r"""You are a helpful AI Assistant. First, think through the reasoning inside <think>...</think>.
Then, identify the correct option and present **only the option's letter** in \boxed{}.""".strip()


DATASETS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "question_column": "question",
        "solution_column": "answer",
    },
    "math": {
        "path": "EleutherAI/hendrycks_math",
        "question_column": "problem",
        "solution_column": "solution",
    },
    "aqua_rat": {
        "path": None,
        "question_column": "question_with_choices",
        "solution_column": "correct",
    },
}


def process_aqua_rat_row(example):
    question = example["question"]
    options_list = example["options"]
    options_str = "\n".join(options_list)

    QUESTION_PROMPT = f"""Solve the following multiple-choice problem.
Question:
    {question}
Options:
    {options_str}"""
    return {"question_with_choices": QUESTION_PROMPT}


def process_aqua_rat(split):
    data = load_dataset("json", data_files=f"data/aqua_rat/{split}.json")
    data = data.map(process_aqua_rat_row)
    data[split].to_json(f"data/aqua_rat/processed/{split}.json")


def make_conversation(
    tokenizer, question_column, solution_column, system_prompt, example
):
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example[question_column]},
    ]
    prompt = tokenizer.apply_chat_template(
        conversation,
        continue_final_message=False,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt, "solution": example[solution_column]}


def get_dataset(name, split, subset, tokenizer):
    data_params = DATASETS[name]
    if name == "aqua_rat":
        process_aqua_rat(split)
        data = load_dataset(
            "json", split=split, data_files=f"data/aqua_rat/processed/{split}.json"
        )
    else:
        data = load_dataset(data_params["path"], subset, split=split)
    system_prompt = (
        SYSTEM_PROMPT_MULTIPLE_CHOICE if name == "aqua_rat" else SYSTEM_PROMPT
    )
    dataset = data.map(
        partial(
            make_conversation,
            tokenizer,
            data_params["question_column"],
            data_params["solution_column"],
            system_prompt,
        )
    )
    return dataset


def load_model(model_name):
    llm = LLM(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, tokenizer


def request_output_to_dict(request_output, solution, level=None):
    ret = {
        "prompt": request_output.prompt,
        "solution": solution,
        "completions": [
            {
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
            }
            for completion in request_output.outputs
        ],
    }
    if level is not None:
        ret["level"] = level
    return ret


def get_math_results(o, solutions, batch):
    difficulty_level = batch["level"]
    results = [
        request_output_to_dict(request_output, solution, level)
        for (request_output, solution, level) in zip(o, solutions, difficulty_level)
    ]
    return results


def generate_answers(llm, dataset, res_folder, dataset_name):
    temp = 0.9
    max_tokens = 32_728
    sampling_params = SamplingParams(
        n=1, temperature=temp, max_tokens=max_tokens, top_p=0.95, top_k=20, min_p=0
    )

    batch_size = min(100_000, len(dataset))

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        prompts = [prompt for prompt in batch["prompt"]]
        solutions = [solution for solution in batch["solution"]]
        print("generating batch", i, "of size", len(prompts))
        o = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
        if dataset_name == "math":
            results = get_math_results(o, solutions, batch)
        else:
            results = [
                request_output_to_dict(request_output, solution)
                for (request_output, solution) in zip(o, solutions)
            ]

        with open(f"{res_folder}/batch_{i}.json", "w") as f:
            json.dump(results, f, indent=2)


def to_dir_name(name: str) -> str:
    name = name.replace("/", "_").replace(".", "_")
    return re.sub(r"[^\w\-]", "_", name)


def main():
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    split = sys.argv[3]
    subset = sys.argv[4]
    res_folder = f"math_gen_output/{to_dir_name(model_name)}/{to_dir_name(dataset_name)}/{split}/{subset}"
    os.makedirs(res_folder, exist_ok=True)

    llm, tokenizer = load_model(model_name)
    dataset = get_dataset(dataset_name, split, subset, tokenizer)
    print(
        f"Loaded model: {model_name}, dataset: {dataset}, split: {split}, subset: {subset}"
    )
    generate_answers(llm, dataset, res_folder, dataset_name)


if __name__ == "__main__":
    main()
