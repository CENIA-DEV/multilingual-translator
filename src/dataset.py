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

from typing import Any, Optional

from torch.utils.data import Dataset

from tokenizer_wrapper import TokenizerWrapper
from utils import load_jsonl


class Multidirectional(Dataset):
    def __init__(
        self,
        file: str,
        tokenizer: Any,
        max_length: int = 200,
        maps: Optional[dict] = None,
        is_val: bool = False,
        directions: Optional[list[str]] = None,
        direction_separator: str = "-",
        ignore: Optional[list[str]] = None,
    ):
        self.data = None
        self.data = load_jsonl(file)
        self.ignore = set(ignore) if ignore is not None else ignore
        self.is_val = is_val
        if self.ignore is not None:
            prev = len(self.data)
            new_data = []
            for d in self.data:
                keys = set(d.keys())
                if self.is_val:
                    new_data.append(
                        {k: v for k, v in d.items() if k not in self.ignore}
                    )
                else:
                    if len(keys & self.ignore) > 0:
                        continue
                    new_data.append(d)

            self.data = new_data
            new = len(self.data)
            diff = prev - new
            if diff > 0:
                print(f"Ignoring {diff} pairs.")

        self.direction_separator = direction_separator

        self.max_length = max_length
        self._tokenizer_args = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": self.max_length,
            "padding": "max_length",
        }
        self.tokenizer = TokenizerWrapper(tokenizer, **self._tokenizer_args)
        self.maps = maps
        if self.is_val:
            if directions is None:
                directions = self._get_directions()

            self.directions = directions

    def _get_directions(self):
        keys = set()
        for d in self.data:
            for k in d.keys():
                keys.add(k)

        directions = []
        for source in keys:
            for target in keys:
                if source != target:
                    directions.append(f"{source}{self.direction_separator}{target}")

        return directions

    def __getitem__(self, idx):
        if not self.is_val:
            length = len(self.data)
            new_idx = idx % length
            keys = list(self.data[new_idx].keys())
            if idx < length:
                keys.reverse()

            lang1, lang2 = keys

            src_lang = lang1 if self.maps is None else self.maps[lang1]
            tgt_lang = lang2 if self.maps is None else self.maps[lang2]

            text_from = self.data[new_idx][lang1]
            text_to = self.data[new_idx][lang2]

            inputs = self.tokenizer(
                text_from, src_lang=src_lang, tgt_lang=tgt_lang, target_text=text_to
            )

            return inputs
        else:
            return_dict = dict()
            for direction in self.directions:
                source, target = direction.split(self.direction_separator)
                # ignore directions
                if source not in direction or target not in direction:
                    continue

                # check for source and target in data since it may not
                # contain these keys
                if source not in self.data[idx] or target not in self.data[idx]:
                    continue

                src_lang = source if self.maps is None else self.maps[source]
                tgt_lang = target if self.maps is None else self.maps[target]

                text_from = self.data[idx][source]
                text_to = self.data[idx][target]

                inputs = self.tokenizer(
                    text_from, src_lang=src_lang, tgt_lang=tgt_lang, target_text=text_to
                )

                return_dict[direction] = inputs

            return return_dict

    def __len__(self):
        # For training, we must have pairs, so we duplicate the dataset to train
        # in both/all directions.

        # For validation, we must have all translations in all directions.
        return len(self.data) * (1 if self.is_val else 2)
