#The license for the orginal trust score code, produced as part of Google research:
#
#Copyright 2018 Google LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import trustscore
import trustscore_evaluation
import numpy as np
import matplotlib.pyplot as plt
import keras

# Import the dataset
from sklearn import datasets
wine = datasets.load_wine()
X_wine = wine.data
y_wine = wine.target
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target


datasets = [(X_wine, y_wine), (X_digits, y_digits)]
dataset_names = ["Wine", "Digits"]

from sklearn.linear_model import LogisticRegression
# Train logistic regression on digits.
model = LogisticRegression()
model.fit(X_digits[:1300], y_digits[:1300])
# Get outputs on testing set.
y_pred = model.predict(X_digits[1300:])
# Initialize trust score.
trust_model = trustscore.TrustScore()
trust_model.fit(X_digits[:1300], y_digits[:1300])
# Compute trusts score, given (unlabeled) testing examples and (hard) model predictions.
trust_score = trust_model.get_score(X_digits[1300:], y_pred)

# print(trust_score) # appears to print the trust scores for each point in the inputted dataset

# Creates graphs about how correct the preduction is for the wine dataset via logistic regression
for dataset_idx,  dataset_name in enumerate(dataset_names):
  extra_plot_title = dataset_name + " | Logistic Regression | Predict Correct"
  percentile_levels = [0 + 0.5 * i for i in range(200)]
  signal_names = ["Trust Score"]
  signals = [trustscore.TrustScore()]
  trainer = trustscore_evaluation.run_logistic
  X, y = datasets[dataset_idx]
  trustscore_evaluation.run_precision_recall_experiment_general(X,
                                                                y,
                                                                n_repeats=10,
                                                                percentile_levels=percentile_levels,
                                                                trainer=trainer,
                                                                signal_names=signal_names,
                                                                signals=signals,
                                                                extra_plot_title=extra_plot_title,
                                                                skip_print=True,
                                                                predict_when_correct=True)


# Graph about the points in the Wine dataset that were classified incorrectly, broken down by the model confidence and the trust score

for dataset_idx,  dataset_name in enumerate(dataset_names):
  extra_plot_title = dataset_name + " | Logistic Regression | Predict Incorrect"
  percentile_levels = [70 + 0.5 * i for i in range(60)]
  signal_names = ["Trust Score"]
  signals = [trustscore.TrustScore()]
  trainer = trustscore_evaluation.run_logistic
  X, y = datasets[dataset_idx]
  trustscore_evaluation.run_precision_recall_experiment_general(X,
                                                                y,
                                                                n_repeats=10,
                                                                percentile_levels=percentile_levels,
                                                                trainer=trainer,
                                                                signal_names=signal_names,
                                                                signals=signals,
                                                                extra_plot_title=extra_plot_title,
                                                                skip_print=True)



# References:

# Trust Scores functions from the Google Scholars paper: https://github.com/google/TrustScore
