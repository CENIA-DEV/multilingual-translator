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
