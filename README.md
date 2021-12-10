# GPT2 Classifier

This repository shows how to use GPT2 model in classification problem. 

## Data

Here you can see model trained to detect emotions in text. Dataset used to train it comes from Kaggle: [Emotions in text dataset](https://www.kaggle.com/ishantjuyal/emotions-in-text).

## Executing

Example of training the model and running it to predict labels on new data is shown in ```demo.ipynb``` notebook.

## Things to do

- data preprocessing,
- hyperparameters tuning,
- stop using model from the last epoch: it is not always the best. Possible solution: save model for all epochs and then choose the best one according to its results (eg. loss),
- add weights to the loss function to deal with imbalanced dataset.