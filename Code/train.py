import os

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction, \
    trainer_utils

import utils
from config import CHECKPOINT_DIRECTORY

assert __name__ == '__main__', 'Cannot be invoked as a module'

utils.init_random_seed()
# MODEL_NAME = 'bert-base-cased'
# MODEL_NAME = 'gpt2'
# MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
# MODEL_NAME = 'openai-gpt'
MODEL_NAME = 'distilbert-base-uncased'
os.environ["WANDB_PROJECT"] = "Movie- Good or Bad Predictor"  # name of the W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

dataset = load_dataset('csv', data_files='movies_training_validation_data.csv')['train']
split_dataset = dataset.train_test_split(test_size=1/9)
tokenizer = utils.get_tokenizer(MODEL_NAME)

processed_dataset = split_dataset.map(utils.get_record_processor(tokenizer))
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(utils.device)
utils.prepare_model(model, MODEL_NAME, tokenizer)
output_directory = f'{CHECKPOINT_DIRECTORY}/{MODEL_NAME}'
training_arguments = TrainingArguments(
    output_dir=output_directory,
    evaluation_strategy='steps',
    logging_steps=1,
    report_to=['wandb'],
    save_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    save_total_limit=5,
)


def compute_metrics_from_eval_prediction(eval_prediction: EvalPrediction) -> dict:
    logits, labels = eval_prediction
    predictions = logits.argmax(axis=-1)
    return utils.compute_metrics(labels, predictions)


trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['test'],
    compute_metrics=compute_metrics_from_eval_prediction,
)
trainer.train(resume_from_checkpoint=trainer_utils.get_last_checkpoint(output_directory))
