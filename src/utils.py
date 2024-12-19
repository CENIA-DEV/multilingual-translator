# Copyright 2024 Centro Nacional de Inteligencia Artificial (CENIA, Chile). All rights reserved.
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
from typing import Any, Union

from transformers import T5TokenizerFast


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
    assert isinstance(
        tokenizer, (T5TokenizerFast)
    ), f"Currently, {type(tokenizer)} does not support adding new languages."

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
