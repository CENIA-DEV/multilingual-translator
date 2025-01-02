# Copyright 2024 Centro Nacional de Inteligencia Artificial (CENIA, Chile).
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from collections import defaultdict
from typing import Any, Union

import torch
from transformers import NllbTokenizerFast


def load_jsonl(path: str) -> list[dict]:
    """
    Load a JSONL file with any formatting style.

    Args:
        `path` (str):
            Path to the file.

    Returns:
        `list`: A list of dictionaries.
    """
    try:
        data = json.load(open(path))
    except json.decoder.JSONDecodeError:
        data = [json.loads(line) for line in open(path)]

    return data


def add_new_languages(tokenizer: Any, model: Any, languages: Union[dict, list]):
    """
    Modify a tokenizer and a model inplace to add new tokens. This only
    works for mT5 models, since NLLB already has language tokens integrated. If
    a NLLB tokenizer/model is given, this process will be ignored.

    Args:
        `tokenizer` (`Any`):
            Tokenizer to modify.
        `model` (`Any`):
            Model to modify.
        `maps` (`dict`):
            Dictionary containing languages as values.
    """
    if len(languages) == 0:
        return

    vocab = tokenizer.get_vocab()
    if isinstance(languages, dict):
        languages = list(set(languages.values()))

    languages = [lang for lang in languages if lang not in vocab]
    if len(languages) > 0:
        print(f"Adding: {languages}")
        tokenizer.add_special_tokens({"additional_special_tokens": languages})
        model.resize_token_embeddings(len(tokenizer))


def copy_from_tokens(
    tokenizer: Any, model: Any, new_tokens: list[str], copy_tokens: list[str]
):
    assert len(new_tokens) == len(
        copy_tokens
    ), "'new_tokens' and 'copy_tokens' must be the same size."
    vocab = tokenizer.get_vocab()

    shared = (
        model.model.shared if isinstance(tokenizer, NllbTokenizerFast) else model.shared
    )

    for new_token, copy_token in zip(new_tokens, copy_tokens):
        if copy_token == "_":
            continue

        orig_embedding = shared.weight.data[vocab[copy_token]]
        shared.weight.data[vocab[new_token]] = orig_embedding.clone()
        print(f"Initialized embedding for '{new_token}' as '{copy_token}'.")


def direction_map_collator(batch: list[dict]) -> dict:
    """
    Collates returns of `__getitem__` in `Dataset` for translation directions. The key
    difference between this collator and the default PyTorch's collator is that this
    function handles `None` objects by not appending them in the resulting batch. If
    some elements are `None` in the batch, the resulting batch will be of size
    `batch_size - none_objects` for those keys where a value of `None` was present.
    """
    return_dict = defaultdict(dict)

    for data in batch:
        for k1, v1 in data.items():
            if v1 is not None:
                for k2, v2 in data[k1].items():
                    if v2 is not None:
                        if k2 not in return_dict[k1]:
                            return_dict[k1][k2] = []
                        return_dict[k1][k2].append(v2)

    return_dict = dict(return_dict)
    for k in return_dict.keys():
        for k2, v2 in return_dict[k].items():
            stack = torch.stack if isinstance(v2[0], torch.Tensor) else torch.tensor
            return_dict[k][k2] = stack(v2)

    return return_dict


def get_directions(data: list[dict], direction_separator: str):
    keys = set()
    for d in data:
        for k in d.keys():
            keys.add(k)

    directions = []
    for source in keys:
        for target in keys:
            if source != target:
                directions.append(f"{source}{direction_separator}{target}")

    return directions
