import pandas

import config
import utils

training_test_split = 0.9

utils.init_random_seed()
shuffled_csv_data = pandas.read_csv("movies_data.csv").sample(frac=1, random_state=config.RANDOM_SEED)
split_idx = round(len(shuffled_csv_data) * 0.9)
training_and_validation_data: pandas.DataFrame = shuffled_csv_data.iloc[:split_idx]
test_data: pandas.DataFrame = shuffled_csv_data.iloc[split_idx: -1]
training_and_validation_data.to_csv('movies_training_validation_data.csv', index=False)
test_data.to_csv('movies_test_data.csv', index=False)
