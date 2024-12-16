import os

from accmt import MLFlow, Monitor, Trainer, set_seed
from dotenv import load_dotenv
from transformers import AutoConfig, NllbTokenizerFast, T5TokenizerFast

from arguments import TrainingArguments, get_config
from dataset import Multidirectional
from metrics import get_metric_modules
from model import Translator

load_dotenv()
TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]

set_seed(41941)
train_config = get_config()
args = TrainingArguments(train_config)

config = AutoConfig.from_pretrained(args.MODEL)
config.dropout = args.DROPOUT if args.DROPOUT is not None else config.dropout

tokenizer_type = NllbTokenizerFast if args.MODEL_TYPE == "nllb" else T5TokenizerFast
tokenizer = tokenizer_type.from_pretrained(args.TOKENIZER, cache_dir="cache")

module = Translator(args, config, tokenizer)
dataset_kwargs = {
    "tokenizer": tokenizer,
    "maps": args.MAPS,
    "max_length": args.MAX_LENGTH,
    "ignore": args.IGNORE,
}
train_dataset = Multidirectional(args.TRAIN_DATASET, **dataset_kwargs)
val_dataset = Multidirectional(
    args.VALIDATION_DATASET, is_val=True, directions=args.DIRECTIONS, **dataset_kwargs
)

trainer = Trainer(
    hps_config=train_config,
    track_name=args.TRACK_NAME,
    model_path=args.MODEL_PATH,
    log_with=MLFlow,
    logging_dir=TRACKING_URI,
    log_every=args.LOG_EVERY,
    resume=args.RESUME,
    compile=args.COMPILE,
    evaluate_every_n_steps=args.EVALUATE_EVERY_N_STEPS,
    run_name=args.RUN_NAME,
    metrics=get_metric_modules(val_dataset.directions, tokenizer),
    monitor=Monitor(learning_rate=True),
    eval_when_start=False,
)

trainer.fit(module, train_dataset, val_dataset)
