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
