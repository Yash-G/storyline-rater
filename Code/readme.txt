Introduction
---------------

This is a project to classify movies as good or bad using movie title, storyline, and budget

Installation
----------------

Please ensure you have a Windows machine with GPU

1. Install Nvidia Cuda- https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
2. Install Anaconda- https://docs.anaconda.com/free/anaconda/install/windows/
3. Search for the Anaconda terminal in the Windows search bar
4. Right-click and select "Run as administrator"
4. Change to the project directory
5. Enable Powershell script execution using the following command-

            powershell -command "Set-ExecutionPolicy Unrestricted"

6. Run uninstall.ps1 from Powershell to ensure a clean state

            powershell -command "& .\uninstall.ps1"

7. Then run-

            conda create --name AMLProject -y
            conda activate AMLProject
            powershell -command "& .\setup.ps1"


Code Structure
------------------


movies_data.csv
Contains the scraped movie storyline, budget, and revenue data from Wikipedia. This will be fed to the model.

movies_metadata.csv
Provides with fallbacks and metadata which is used to generate movies_data.csv file.
Downloaded from https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv

scrapper.py
The scraper generates movies_metadata.csv file by scraping storyline, budget and revenue data from Wikipedia.
In order to see it in action, delete movies_metadata.csv and replace the contents of previous_run.txt with 0 before running it.

previous_run.txt
The scrapper fails whenever it sees a new page format. The previous_run.txt file contains the count of rows processed
up till the previous run. This allows the scrapper to resume from where it left off after it has been adapted for the new format.

missed_rows.txt
This is not being used currently. If the scraper failed to locate the data, then it used to simply skip that row. This file
used to contain the indexes of those rows.

rows_to_skip.txt
Contains the sequence index of the records that the scrapper should skip. The primary candidates are those movies whose Wikipedia
pages have a very different / complicated format for the scrapper to accommodate for.

config.py
Project-wide configuration. Please be careful while changing the values. Check out the file for the details.

wandb/
"Weights & Biases" is a third party service provider (https://wandb.ai/site) that I am using to visualize my training data.
Their wandb agent syncs my training logs to their server which is then analyzed and shown on their dashboard.
This directory contains the local cache of that data.

evaluation_results/
Contains the charts and metrics demonstrating the performance of the models. The sub-folders are named using a compressed
representation of the model name and variant. The -w- and -wo- stand for with and without, and is prefixed to 'budget' to indicate
whether budget was considered a feature for the prediction.

train.py
Contains the fine-tuning script. Run normally, however, change the MODEL_NAME variable in it for training on different models.
This script also resumes from the last execution if found.

training_checkpoints/ and training_checkpoints_with_budget/
Contains the checkpoint directories for different models

movie_data_splitter.py
Splits the dataset CSV file movie_data.csv into two CSV files- one containing the training data and the other containing test data

movies_training_validation_data.csv
Contains the training and validation part of movie_data.csv

movies_test_data.csv
Contains the test part of movie_data.csv

test_predictor.py
Runs the trained model on the test data in movie_test_data.csv. No need to change any variable in code.
The script will ask you for the model.

predict.py
This is the script that I will be running to determine whether a movie is worth watching :)
Code-side variable change not required for input.

plot_dataset_graphs.py
Gives a graphical overview of data distribution and logs various dataset metrics

utils.py
Module containing code shared amongst multiple scripts

setup.ps1
Powershell script that installs all the required libraries

uninstall.ps1
Cleanup script
