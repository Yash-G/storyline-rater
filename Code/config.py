# Represents missing values in the CSV files our system generates.
NA_STRING = 'NA'

# I previously wanted to use Hindi movies as well for the training, however, there are very few Hindi
# movie listings in the metadata and my sequential scrapper failed to reach the first one.
# So this is effectively equal to ['en'].
MY_LANGUAGES = ['en', 'hi']

RANDOM_SEED = 123456

# Change this to training_checkpoints_with_budget when enabling IS_BUDGET_A_FEATURE below
CHECKPOINT_DIRECTORY = 'training_checkpoints_with_budget'

# Controls whether the budget is used for the classification
IS_BUDGET_A_FEATURE = True
