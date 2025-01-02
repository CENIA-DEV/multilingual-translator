# Multilingual Translator
This repository exposes the code to train multilingual translators using either NLLB or mT5 models.

# How to train
Inside `src`, there is a `train.py` file that you can call as a command providing a configuration file.

To train on one single CUDA device:
```bash
python train.py <config-file>
```

To train on multiple CUDA devices, we use [https://github.com/ghanvert/AcceleratorModule](accmt):
```bash
accmt launch train.py <config-file>
```

# Configuration file
The configuration file consists of different settings to adjust your training. This consists of a YAML file with the following keys:
| Key                      | Definition                                                          |
|--------------------------|---------------------------------------------------------------------|
| `track_name`             | Track/Experiment name on MLFlow.                                    |
| `run_name`               | Run name in experiment on MLFlow.                                   |
| `log_every`              | Log every N steps to MLFlow.                                        |
| `evaluate_every_n_steps` | Do evaluation every N steps.                                        |
| `model`                  | Path to model to finetune.                                          |
| `model_path`             | Output model path where to save best model and progress.            |
| `model_type`             | Model type: **nllb** or **mt5**.                                    |
| `tokenizer`              | Path to tokenizer to use.                                           |
| `compile`                | Compile model for training.                                         |
| `dropout`                | Dropout rate.                                                       |
| `rdrop`                  | Enable RDROP regularization technique.                              |
| `rdrop_alpha`            | RDROP alpha value.                                                  |
| `label_smoothing`        | Label Smoothing value.                                              |
| `max_length`             | Max length for model inputs/outputs during training.                |
| `train_dataset`          | Train JSON dataset path.                                            |
| `validation_dataset`     | Validation JSON dataset path.                                       |
| `maps`                   | Map JSON keys in dataset to the corresponding language tokens.      |
| `directions`             | Directions to train as a list. Example: `eng-spa`, `spa-eng`, etc.  |
| `resume`                 | Resume training. If not specified, this will be done automatically. |
| `hps`                    | Hyperparameters for training. See `example_config.yaml`.            |

See `examples/example_config.yaml` for more details.

# Dataset format
Here we show the dataset format both for train and validation.

## Train dataset
This must be a JSON file with a list of only pairs. See `examples/example_train_dataset.jsonl`.

## Validation dataset
This must be a JSON file with a list of a single sentence with its various translations. See `examples/example_validation_dataset.jsonl`.

Only the `directions` in the configuration file will be evaluated. Other ones will be ignored.

# MLFlow Setup
You can setup MLFlow locally:
```bash
mlflow server --host=localhost --port=5000
```
Then you can go to your browser: https://localhost:5000/

Also, you must have a `.env` file in this directory (`multilingual-translator/`) with the `MLFLOW_TRACKING_URI` variable defined. This can be `localhost:5000` or any other address to your MLFlow server.

# mT5
## Training
Before training mT5 models, you need to make sure to add language tokens to both the tokenizer and the model's embeddings. For this, you can use the script `convert_model.py`:
```bash
python convert_model <path-to-mt5-model> --new-tokens=<list-of-tokens> --from-tokens=<list-of-previous-tokens> -O <output-path>
```

You will add new tokens via `new-tokens`. If you want to copy embeddings from previous tokens to the new ones, you can use `from-tokens`. Both lists of tokens must have the same size. If you want to ignore the process of copying embeddings to a specific token, ignore them using `_` character in `from-tokens`.

## Inference
Example:
```python
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM

tokenizer = T5TokenizerFast.from_pretrained("path-to-your-model-or-tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained("path-to-your-model")

def translate(sentence: str, translate_from="spa_Latn", translate_to="eng_Latn") -> str:
    inputs = tokenizer(translate_from + sentence, return_tensors="pt")
    result = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(translate_to))
    decoded = tokenizer.batch_decode(result, skip_special_tokens=True)[0]
    return decoded
```

# NLLB
## Training
For language tokens, make sure to check available languages in [https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/special_tokens_map.json](NLLB's tokenizer).

## Inference
```python
from transformers import NllbTokenizerFast, AutoModelForSeq2SeqLM

tokenizer = NllbTokenizerFast.from_pretrained("path-to-your-model-or-tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained("path-to-your-model")

def translate(sentence: str, translate_from="spa_Latn", translate_to="eng_Latn") -> str:
    tokenizer.src_lang = translate_from
    tokenizer.tgt_lang = translate_to

    inputs = tokenizer(sentence, return_tensors="pt")
    result = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(translate_to))
    decoded = tokenizer.batch_decode(result, skip_special_tokens=True)[0]
    return decoded
```
