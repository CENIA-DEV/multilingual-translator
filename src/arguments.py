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

from argparse import ArgumentParser
from typing import Optional

import yaml


class TrainingArguments:
    """
    Get training arguments from a YAML or JSON file.

    The following arguments are defined inside this class:
        RESUME (`bool`, *optional*):
            If set to `True` or `False`, it forces to resume training. If not defined,
            automatically detects if a checkpoint exists.
        MODEL (`str`):
            Model to be finetuned. This should be a path to a directory containing the
            model itself in HuggingFace's format.
        MODEL_TYPE (`str`, *optional*, defaults to `nllb`):
            Model type. Available types are: `nllb` and `mt5`.
        TOKENIZER (`str`):
            Tokenizer path.
        MODEL_PATH (`str`):
            Output model path where to save the model.
        TRAIN_DATASET (`str`):
            Train dataset path.
        VALIDATION_DATASET (`str`):
            Validation dataset path.
        TRACK_NAME (`str`):
            Track name in MLFlow.
        RUN_NAME (`str`, *optional*):
            Run name inside `TRACK_NAME` in MLFlow. If not defined, a name will be
            generated for this run.
        DROPOUT (`float`, *optional*):
            Set a custom Dropout value for the model.
        RDROP (`bool`):
            Applies RDROP regularization technique (https://arxiv.org/abs/2106.14448).
        RDROP_ALPHA (`float`, *optional*, defaults to `5`):
            Applies an alpha factor to the RDROP regularization.
        LABEL_SMOOTHING (`float`, *optional*):
            Applies Label Smoothing regularization technique
            (https://arxiv.org/pdf/1512.00567).
        MAPS (`dict`, *optional*):
            Dictionary key to corresponding language token.
        DIRECTIONS (`list`, *optional*):
            List of directions.
        LOG_EVERY (`int`, *optional*, defaults to `10`):
            Log train loss to MLFlow every N steps.
        EVALUATE_EVERY_N_STEPS (`int`, *optional*):
            Evaluate every N steps.
        COMPILE (`bool`, *optional*, defaults to `False`):
            Compile model.
        IGNORE (`set`, *optional*):
            Ignore certain pairs.

    Args:
        path (`str`):
            Path to the YAML or JSON file.
    """

    _mandatory_keys = {
        "model",
        "tokenizer",
        "model_path",
        "track_name",
        "train_dataset",
        "validation_dataset",
    }

    def __init__(self, path: str):
        self._data: dict = yaml.safe_load(open(path))
        self._check_arguments()

        self.RESUME: Optional[bool] = self._data.get("resume")
        self.MODEL: str = self._data.get("model")
        self.MODEL_TYPE: str = self._data.get("model_type", "nllb").lower()
        self.TOKENIZER: str = self._data.get("tokenizer")
        self.MODEL_PATH: str = self._data.get("model_path")
        self.TRACK_NAME: str = self._data.get("track_name")
        self.RUN_NAME: Optional[str] = self._data.get("run_name")
        self.MAX_LENGTH: int = self._data.get("max_length", 200)
        self.DROPOUT: Optional[float] = self._data.get("dropout")
        self.RDROP: bool = self._data.get("rdrop", False)
        self.RDROP_ALPHA: float = self._data.get("rdrop_alpha", 5)
        self.LABEL_SMOOTHING: Optional[float] = self._data.get("label_smoothing")
        self.TRAIN_DATASET: str = self._data.get("train_dataset")
        self.VALIDATION_DATASET: str = self._data.get("validation_dataset")
        self.MAPS: dict = self._data.get("maps")
        self.DIRECTIONS: list = self._data.get("directions")
        self.LOG_EVERY: int = self._data.get("log_every", 10)
        self.EVALUATE_EVERY_N_STEPS: Optional[int] = self._data.get(
            "evaluate_every_n_steps"
        )
        self.COMPILE: Optional[bool] = self._data.get("compile", False)
        self.IGNORE: Optional[list[str]] = self._data.get("ignore")

    def _check_arguments(self):
        keys = set(self._data.keys())
        intersection = keys & self._mandatory_keys

        if intersection != self._mandatory_keys:
            missing_keys = list(self._mandatory_keys - intersection)
            raise ValueError(f"Missing keys: {missing_keys}")


def get_config():
    parser = ArgumentParser(
        description="Train a translation model with a given configuration file."
    )

    parser.add_argument("config", type=str, help="Configuration YAML or JSON file.")
    args = parser.parse_args()

    return args.config
