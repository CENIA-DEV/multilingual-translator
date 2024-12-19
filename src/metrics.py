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

from typing import Any

import evaluate
from accmt.metrics import Metric

_bleu_module = evaluate.load("bleu")


class BLEU(Metric):
    tokenizer = None

    def get_texts(self, tensor) -> list[str]:
        return self.tokenizer.batch_decode(tensor, skip_special_tokens=True)

    def compute(self, predictions, references):
        predictions = self.get_texts(predictions)
        references = self.get_texts(references)

        results = _bleu_module.compute(predictions=predictions, references=references)

        return {self.main_metric: results["bleu"]}


def get_metric_modules(directions: list[str], tokenizer: Any) -> list[Metric]:
    modules = []
    for direction in directions:
        module = BLEU(f"bleu_{direction}")
        module.tokenizer = tokenizer
        modules.append(module)

    return modules
