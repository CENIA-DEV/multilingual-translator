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

from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast, T5TokenizerFast

from utils import add_new_languages, copy_from_tokens

parser = ArgumentParser(description="Convert model and tokenizer.")
parser.add_argument("model", type=str, help="Path to the model.")
parser.add_argument(
    "--tokenizer",
    type=str,
    help="Path to a tokenizer. If not provided, 'model' path will be used.",
)
parser.add_argument(
    "--new-tokens", type=str, nargs="+", required=True, help="New tokens to add."
)
parser.add_argument(
    "--from-tokens",
    type=str,
    nargs="+",
    help="Copy embeddings from previous tokens to the new tokens. Use '_' to ignore.",
)
parser.add_argument("--type", type=str, required=True, help="Model type.")
parser.add_argument("--output", "-O", required=True, type=str, help="Output path.")
args = parser.parse_args()

available_types = ["mt5", "nllb"]
model_type = args.type.lower()
assert model_type in available_types, f"Available models types are: {available_types}"

model_path = args.model
tokenizer_path = model_path if args.tokenizer is None else args.tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer_type = T5TokenizerFast if model_type == "mt5" else NllbTokenizerFast
tokenizer = tokenizer_type.from_pretrained(tokenizer_path)

add_new_languages(tokenizer, model, args.new_tokens)
if args.from_tokens is not None:
    copy_from_tokens(tokenizer, model, args.new_tokens, args.from_tokens)
model.save_pretrained(args.output, safe_serialization=False)
tokenizer.save_pretrained(args.output)
