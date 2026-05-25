import glob
import json
import os

import numpy as np
import pandas as pd
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from tqdm.auto import tqdm

OUTPUT_DIR = "../data"


def parse_completion(completion):
    return parse(
        completion,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=False,
                    boxed="all",
                    units=True,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )


MODELS = [
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B",
    "agentica-org_DeepScaleR-1_5B-Preview",
]

DATASETS = {"aqua_rat": 0, "gsm8k": 1, "math": 2}


def parse_aqua_results():
    for model in MODELS:
        for split in ["test", "train"]:
            path = f"math_gen_output/{model}/aqua_rat/{split}/batch_0.json"
            with open(path, "r") as f:
                data = json.load(f)
            res = []
            for q in data:
                solution = q["solution"]
                completion = parse_completion(q["completions"][0]["text"])
                res.append(verify(solution, completion))
            df = pd.DataFrame({"y": np.array(res).astype(int)})
            os.makedirs(f"data/math/{model}/aqua_rat/{split}", exist_ok=True)
            df.to_parquet(f"data/math/{model}/aqua_rat/{split}/results.parquet")


def parse_gsm8k_results():
    for model in MODELS:
        for split in ["test", "train"]:
            path = f"math_gen_output/{model}/gsm8k/{split}/main/batch_0.json"
            with open(path, "r") as f:
                data = json.load(f)
            res = []
            for q in data:
                solution = q["solution"].split("#### ")[-1]
                completion = parse_completion(q["completions"][0]["text"])
                res.append(verify(solution, completion))
            df = pd.DataFrame({"y": np.array(res).astype(int)})
            os.makedirs(f"data/math/{model}/gsm8k/{split}", exist_ok=True)
            df.to_parquet(f"data/math/{model}/gsm8k/{split}/results.parquet")


def parse_math_result():
    for model in MODELS:
        for split in ["test", "train"]:
            base_dir = f"math_gen_output/{model}/math/{split}"
            pattern = os.path.join(base_dir, "**", "*.json")
            files = glob.glob(pattern, recursive=True)
            files = sorted(files)
            data = []
            for fname in files:
                with open(fname, "r") as f:
                    loaded = json.load(f)
                data.extend(loaded)

            res = []
            for q in data:
                solution = parse_completion(q["solution"])
                completion = parse_completion(q["completions"][0]["text"])
                res.append(verify(solution, completion))

            df = pd.DataFrame({"y": np.array(res).astype(int)})
            out_dir = f"{OUTPUT_DIR}/math/{model}/math/{split}"
            os.makedirs(out_dir, exist_ok=True)
            df.to_parquet(f"{out_dir}/results.parquet")


REFERENCE_MODEL = "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B"


def _load_math_data(model, split):
    """Load and return all math data for a given model and split."""
    base_dir = f"math_gen_output/{model}/math/{split}"
    pattern = os.path.join(base_dir, "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    files = sorted(files)
    data = []
    for fname in files:
        with open(fname, "r") as f:
            loaded = json.load(f)
        data.extend(loaded)
    return data


def _load_gsm8k_data(model, split):
    """Load and return all gsm8k data for a given model and split."""
    path = f"math_gen_output/{model}/gsm8k/{split}/main/batch_0.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data


def _load_aqua_rat_data(model, split):
    """Load and return all aqua_rat data for a given model and split."""
    path = f"math_gen_output/{model}/aqua_rat/{split}/batch_0.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data


def _get_parsed_completion(question):
    """Parse and return the completion from a question."""
    return parse_completion(question["completions"][0]["text"])


def _get_correctness(question, dataset):
    """Determine if the model's answer is correct for the given question."""
    completion = _get_parsed_completion(question)
    if dataset == "aqua_rat":
        solution = question["solution"]
    elif dataset == "gsm8k":
        solution = question["solution"].split("#### ")[-1]
    else:  # math
        solution = parse_completion(question["solution"])
    return verify(solution, completion)


def create_xy_dataframes(reference_model, output_dir):
    """
    Create dataframes with x and y columns for each model, dataset, and split.

    y: binary column indicating if the model was correct on the question
    x: binary column indicating if the model's answer matches the reference model
       (deepseek-ai_DeepSeek-R1-Distill-Qwen-14B)

    Returns a dict: {model: {dataset: {split: DataFrame}}}
    """
    dataset_loaders = {
        "aqua_rat": _load_aqua_rat_data,
        "gsm8k": _load_gsm8k_data,
        "math": _load_math_data,
    }

    results = {}

    for dataset, loader in dataset_loaders.items():
        for split in ["test", "train"]:
            print(f"Processing {dataset}/{split}...")

            # Load reference model data first
            try:
                ref_data = loader(reference_model, split)
            except FileNotFoundError:
                print(f"Warning: Reference model data not found for {dataset}/{split}")
                continue

            # Parse reference model completions
            ref_completions = [_get_parsed_completion(q) for q in ref_data]

            for model in MODELS:
                print(f"Processing {model} on {dataset}/{split}...")
                if model not in results:
                    results[model] = {}
                if dataset not in results[model]:
                    results[model][dataset] = {}

                try:
                    model_data = loader(model, split)
                except FileNotFoundError:
                    print(f"Warning: Data not found for {model}/{dataset}/{split}")
                    continue

                if len(model_data) != len(ref_data):
                    print(
                        f"Warning: Data length mismatch for {model}/{dataset}/{split}"
                    )
                    continue

                y_values = []
                x_values = []

                for i, q in tqdm(enumerate(model_data)):
                    # y: is the model correct?
                    y = _get_correctness(q, dataset)
                    y_values.append(y)

                    # x: does model answer match reference model answer?
                    model_completion = _get_parsed_completion(q)
                    ref_completion = ref_completions[i]
                    x = verify(model_completion, ref_completion)
                    x_values.append(x)

                df = pd.DataFrame(
                    {
                        "x": np.array(x_values).astype(int),
                        "y": np.array(y_values).astype(int),
                    }
                )

                results[model][dataset][split] = df

                # Save to disk
                out_dir = f"{output_dir}/{model}/{dataset}/{split}"
                os.makedirs(out_dir, exist_ok=True)
                df.to_parquet(f"{out_dir}/results_xy.parquet")
                print(f"Saved: {out_dir}/results_xy.parquet")

    return results


if __name__ == "__main__":
    create_xy_dataframes(REFERENCE_MODEL, f"{OUTPUT_DIR}/math_judge")
    parse_math_result()
