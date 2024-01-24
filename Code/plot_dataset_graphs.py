import functools
import os

import numpy
import pandas
import matplotlib.pyplot as plt
import datasets

import config
import utils

csv_data = pandas.read_csv('movies_training_validation_data.csv')
budget = csv_data['budget'].to_numpy(dtype=numpy.float64)
figure, ax = plt.subplots()
ax.hist(budget)
ax.set_xlabel('Budget (in USD)')
ax.set_ylabel('Count')
ax.set_title('Budget Histogram')
labels = csv_data['is_good'].to_numpy(dtype=numpy.int64)
label_cnts = [0, 0]
for label in labels:
    label_cnts[label] += 1
figure, ax = plt.subplots()
ax.bar(x=['Bad', 'Good'], height=label_cnts, color=['Red', 'Green'])
ax.set_title('Bar graph showing count distribution of good & bad movies')
ax.set_ylabel('Count')
# figure.hist(csv_data['is_good'], bins=[0, 0.5, 1])
figure, ax = plt.subplots()
sanitized_budget_array = budget[~numpy.isnan(budget)]
ax.boxplot(sanitized_budget_array)
ax.set_title('Budget Box Plot')

with open('previous_run.txt') as previous_run_file:
    processed_row_count = int(previous_run_file.readline().strip())


def fraction_to_percent_string(frac: float) -> str:
    return utils.get_rounded_string(frac * 100, 2)


data_file_dtype = {'budget': str, 'revenue': str, 'is_story_detailed': str}
movie_metadata_frame = pandas.read_csv('movies_metadata.csv', dtype=data_file_dtype, keep_default_na=False)[:processed_row_count + 1]
movie_metadata_frame = movie_metadata_frame[movie_metadata_frame['original_language'].isin(config.MY_LANGUAGES)]
movie_data_frame = pandas.read_csv('movies_data.csv', dtype=data_file_dtype, keep_default_na=False)
plot_row_count = len(movie_data_frame[movie_data_frame['is_story_detailed'] == 'True'])
overview_row_count = len(movie_data_frame[movie_data_frame['is_story_detailed'] == 'False'])
_, ax = plt.subplots()
ax.pie([plot_row_count, overview_row_count], labels=['Plot', 'Overview'], autopct='%.1f%%')
ax.set_title('Story Attribute Type Distribution')
dataset_stats = {
    'metadata_row_count': len(movie_metadata_frame),
    'data_row_count': len(movie_data_frame),
    'plot_row_count': plot_row_count,
    'overview_row_count': overview_row_count,
    'missing_budget_metadata_row_count': len(movie_metadata_frame[movie_metadata_frame['budget'] == '0']),
    'missing_revenue_metadata_row_count': len(movie_metadata_frame[movie_metadata_frame['revenue'] == '0']),
    'missing_budget_row_count': len(movie_data_frame[movie_data_frame['budget'] == config.NA_STRING]),
    'missing_revenue_row_count': len(movie_data_frame[movie_data_frame['revenue'] == config.NA_STRING]),
    'present_budget_metadata_row_count': len(movie_metadata_frame[movie_metadata_frame['budget'] != '0']),
    'present_revenue_metadata_row_count': len(movie_metadata_frame[movie_metadata_frame['revenue'] != '0']),
    'present_budget_row_count': len(movie_data_frame[movie_data_frame['budget'] != config.NA_STRING]),
    'present_revenue_row_count': len(movie_data_frame[movie_data_frame['revenue'] != config.NA_STRING]),
}
dataset_stats['plot_row_percent'] = fraction_to_percent_string(plot_row_count / len(movie_data_frame))
dataset_stats['overview_row_percent'] = fraction_to_percent_string(overview_row_count / len(movie_data_frame))
dataset_stats['missing_budget_metadata_row_percent'] = fraction_to_percent_string(dataset_stats['missing_budget_metadata_row_count'] / len(movie_metadata_frame))
dataset_stats['missing_revenue_metadata_row_percent'] = fraction_to_percent_string(dataset_stats['missing_revenue_metadata_row_count'] / len(movie_metadata_frame))
dataset_stats['missing_budget_row_percent'] = fraction_to_percent_string(dataset_stats['missing_budget_row_count'] / len(movie_data_frame))
dataset_stats['missing_revenue_row_percent'] = fraction_to_percent_string(dataset_stats['missing_revenue_row_count'] / len(movie_data_frame))
dataset_stats['present_budget_metadata_row_percent'] = fraction_to_percent_string(dataset_stats['present_budget_metadata_row_count'] / len(movie_metadata_frame))
dataset_stats['present_revenue_metadata_row_percent'] = fraction_to_percent_string(dataset_stats['present_revenue_metadata_row_count'] / len(movie_metadata_frame))
dataset_stats['present_budget_row_percent'] = fraction_to_percent_string(dataset_stats['present_budget_row_count'] / len(movie_data_frame))
dataset_stats['present_revenue_row_percent'] = fraction_to_percent_string(dataset_stats['present_revenue_row_count'] / len(movie_data_frame))
print(f'Dataset metrics-\n{dataset_stats}\n')
print('Count by Language:')
print(movie_data_frame.groupby('original_language').size())
print(f'Total count: {len(movie_data_frame)}')

dataset = datasets.load_dataset('csv', data_files='movies_training_validation_data.csv')['train']
seq_len_hist_plotters = []
seq_len_box_plot_plotters = []
model_names = set(model_name for directory in ['training_checkpoints', 'training_checkpoints_with_budget'] for model_name in os.listdir(directory))
for model_name in model_names:
    tokenizer = utils.get_tokenizer(model_name)
    preprocessor = utils.get_record_processor(tokenizer=tokenizer, padding='do_not_pad', truncation=False)

    def extract_seq_length(record):
        input_ids = preprocessor(record)['input_ids']
        return len(input_ids)


    max_sequence_length = tokenizer.model_max_length
    considered_tokens = functools.reduce(lambda acc, record: acc + min(max_sequence_length, extract_seq_length(record)), dataset, 0)
    all_tokens = functools.reduce(lambda acc, record: acc + extract_seq_length(record), dataset, 0)
    considered_token_percentage = fraction_to_percent_string(considered_tokens / all_tokens)
    mapped_dataset = dataset.map(lambda record: record | {'sequence_length': extract_seq_length(record)})
    sequence_lengths = numpy.array(mapped_dataset['sequence_length'])
    truncation_count = len(sequence_lengths[sequence_lengths > max_sequence_length])
    intact_count = len(sequence_lengths) - truncation_count
    dataset_metrics_for_model = {
        'model': model_name,
        'maximum_sequence_length': max_sequence_length,
        'considered_tokens': considered_tokens,
        'all_tokens': all_tokens,
        'considered_token_percentage': considered_token_percentage,
        'truncation_count': truncation_count,
        'intact_count': intact_count,
        'truncation_percent': fraction_to_percent_string(truncation_count / len(sequence_lengths)),
        'intact_percent': fraction_to_percent_string(intact_count / len(sequence_lengths)),
    }
    print(f'Dataset metrics for model: \n{dataset_metrics_for_model}')

    def hist_plotter(model_name=model_name, seq_lengths=sequence_lengths):
        _, ax = plt.subplots()
        ax.set_title(f'{model_name} sequence length histogram')
        ax.hist(seq_lengths)

    def boxplot_plotter(model_name=model_name, seq_lengths=sequence_lengths):
        _, ax = plt.subplots()
        ax.set_title(f'{model_name} sequence length box plot')
        ax.boxplot(seq_lengths)

    seq_len_hist_plotters.append(hist_plotter)
    seq_len_box_plot_plotters.append(boxplot_plotter)

for plotter in seq_len_hist_plotters:
    plotter()
for plotter in seq_len_box_plot_plotters:
    plotter()

plt.show()
