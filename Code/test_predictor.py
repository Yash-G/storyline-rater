import matplotlib.pyplot as plt
import pandas
import sklearn.metrics
import torch
from datasets import load_dataset
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

import utils

assert __name__ == '__main__', 'Cannot be invoked as a module'

dataset = load_dataset('csv', data_files='movies_test_data.csv')['train']
model_output = utils.run_predictions(dataset)
data_frame = pandas.DataFrame(dataset)
data_frame['prediction'] = model_output['prediction']
data_frame['chance'] = model_output['chance']
data_frame.to_csv('movies_test_data_with_predictions.csv', index=False)
is_good_int_tensor = torch.tensor(dataset["is_good"], dtype=torch.int64)
predictions = model_output["prediction"]
metrics = utils.compute_metrics(is_good_int_tensor, predictions)
print('Metrics-', metrics)
RocCurveDisplay.from_predictions(is_good_int_tensor, predictions)
ConfusionMatrixDisplay.from_predictions(is_good_int_tensor, predictions)
plt.show()
