# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
# ]
# ///
"""
Preprocess the dataset for the 'count_rs' environment in parquet format.
"""

import argparse
import os
import random
import string
from collections import defaultdict

import requests
from datasets import Dataset


# Target character we are counting
C = "r"


def is_valid_word(word: str) -> bool:
    return all(c in string.ascii_lowercase for c in word)


SYSTEM_PROMPT = {
    "role": "system",
    "content": f"Your goal is to count the number of times the character {C} appears in the word given by the user. Think step by step before answering. Put your final answer in \\boxed{{answer}} format.",
}


def prepare_dataset(words: list[str], split_name: str) -> Dataset:
    examples = []
    for word in words:
        user_prompt = f"How many times appear the character {C} in the word: {word}?"
        data = {
            "data_source": "online_words",
            "prompt": [
                SYSTEM_PROMPT,
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "env_class": "count_rs",
            "reward_spec": {
                "method": "rule",
                "ground_truth": str(word.count(C)),
            },
            "extra_info": {
                "word": word,
                "split": split_name,
            },
        }
        examples.append(data)

    return Dataset.from_list(examples)


def create_dataset(
    url: str, train_samples: int, test_samples: int
) -> tuple[Dataset, Dataset]:
    """Create a dataset of multiplication problems."""
    words = requests.get(url).text.splitlines()
    words = [word for word in words if is_valid_word(word)]

    freq = defaultdict(list)
    for word in words:
        freq[word.count(C)].append(word)

    total = train_samples + test_samples
    selected_words = []
    remaining_words = total
    reamining_chunks = len(freq.keys())

    for key in reversed(sorted(freq.keys())):
        try_select = remaining_words // reamining_chunks
        k_words = freq[key]
        select = min(try_select, len(k_words))
        remaining_words -= select
        reamining_chunks -= 1
        selected_words.extend(random.sample(k_words, select))

    random.shuffle(selected_words)
    test_samples = len(selected_words) * test_samples // total
    train_samples = len(selected_words) - test_samples

    train_dataset = prepare_dataset(selected_words[:train_samples], "train")
    test_dataset = prepare_dataset(selected_words[train_samples:], "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words.txt",
    )
    parser.add_argument("--output_dir", default="~/data/count_rs")
    parser.add_argument(
        "--train_size", type=int, default=5000, help="Number of training examples"
    )
    parser.add_argument(
        "--test_size", type=int, default=500, help="Number of test examples"
    )

    args = parser.parse_args()

    print("Creating dataset...")
    # Generate datasets
    train_dataset, val_dataset = create_dataset(
        args.url,
        args.train_size,
        args.test_size,
    )

    # Save datasets
    output_dir = args.output_dir
    output_dir = output_dir.replace("~", os.path.expanduser("~"))
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

    print(
        f"Generated {args.train_size} training examples and {args.test_size} test examples"
    )
    print(f"Saved to {output_dir}")
