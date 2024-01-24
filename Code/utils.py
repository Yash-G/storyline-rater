import os
import sys
import warnings
from typing import Protocol

import torch
import transformers
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import BatchEncoding, AutoModelForSequenceClassification, AutoTokenizer

import config


class TokenizerType(Protocol):
    def __call__(self, text, padding: str, truncation: bool) -> BatchEncoding: pass


def prepare_input(title, budget, story, tokenizer: TokenizerType, padding='max_length', truncation=True, **kwargs) -> BatchEncoding:
    if config.IS_BUDGET_A_FEATURE:
        budget_string = f'a budget of {budget}' if budget != config.NA_STRING else 'an unknown budget'
        text = (f"The movie '{title}' was originally made on {budget_string}. "
                f"Here goes the plot/overview--\n\n") + story
    else:
        text = (f"The movie title is '{title}'. "
                f"Here goes the plot/overview--\n\n") + story
    return tokenizer(text, padding=padding, truncation=truncation)


def get_rounded_string(x: float, decimal_places: int) -> str:
    return f'%.{decimal_places}f' % round(x, decimal_places)


def init_random_seed():
    transformers.set_seed(config.RANDOM_SEED)


def get_record_processor(tokenizer: TokenizerType, padding='max_length', truncation=True):
    return lambda record: prepare_input(**record, tokenizer=tokenizer, padding=padding, truncation=truncation) | (
        {'labels': int(record['is_good'])} if 'is_good' in record else {})


def get_tokenizer(model_name: str) -> TokenizerType:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name == 'openai-gpt':
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def prepare_model(model, model_name: str, tokenizer) -> None:
    if model_name == 'openai-gpt':
        model.config.pad_token_id = tokenizer.unk_token_id


device: str
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    warnings.warn('GPU not available!')


def run_predictions(dataset: Dataset):
    model_names = os.listdir(config.CHECKPOINT_DIRECTORY)

    match len(model_names):
        case 0:
            print('Aborting as model data is missing')
            sys.exit()
        case 1:
            print(f'Model {model_names[0]} is the only one detected, hence continuing with it')
            model_idx = 0
        case _:
            for i, model_name in enumerate(model_names):
                print(f'{i + 1}) {model_name}')
            model_idx = int(input('Select a model no.')) - 1

    selected_model_name = model_names[model_idx]
    checkpoint_path = transformers.trainer_utils.get_last_checkpoint(
        f'{config.CHECKPOINT_DIRECTORY}/{selected_model_name}')
    if checkpoint_path is None:
        print(f'Model {selected_model_name} does not have any saved checkpoints, hence aborting.')
        sys.exit()
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = get_tokenizer(selected_model_name)
    prepare_model(model, selected_model_name, tokenizer)
    record_processor = get_record_processor(tokenizer)

    def record_mapper(record):
        processed_record = record_processor(record).convert_to_tensors('pt')
        model_output = model(**{k: v.unsqueeze(dim=0) for k, v in processed_record.items()})
        squeezed_model_output = {key: tensor.squeeze() for key, tensor in model_output.items()}
        logits = squeezed_model_output['logits']
        prediction = logits.argmax(-1)
        chances = logits.softmax(-1) * 100
        chance_string = [get_rounded_string(chance.item(), 1) for chance in chances]
        return squeezed_model_output | {'prediction': prediction, 'chance': chance_string}

    return dataset.map(record_mapper).with_format('torch')


def compute_metrics(y_true, y_pred):
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
    is_correct_array = y_true == y_pred
    prepared_is_correct_array = is_correct_array.double() if isinstance(is_correct_array,
                                                                        torch.Tensor) else is_correct_array
    return {
        'accuracy': prepared_is_correct_array.mean().item(),
        'precision': precision,
        'recall': recall,
        'f1_score': fbeta_score,
    }
