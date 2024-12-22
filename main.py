# import solution.A4codes
import os 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from torch import nn, optim
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from helper import learn, classify

# Run the evaluation
train_path = f"{os.path.dirname(__file__)}\\dataset\\training-dataset.csv"  # Replace with the actual training file path
val_path = f"{os.path.dirname(__file__)}\\dataset\\validation-dataset.csv"     # Replace with the actual validation file path


######################### Testing algorithms ############################
def getMatrices(file_name: str):
  """
  Prepares the dataset from a CSV file, optionally applying data augmentation.

  Args:
      file_name (str): Path to the CSV file.
      augment (bool): Whether to apply data augmentation (only for training data).

  Returns:
      dataset (TensorDataset): The dataset as PyTorch TensorDataset.
      dataloader (DataLoader): DataLoader for batching and shuffling.
  """
  train_data = np.loadtxt(file_name, delimiter=',')
  y = train_data[:, 0]
  X = train_data[:, 1:] / 255.

  return X, y

train_x, train_y = getMatrices(train_path)
test_x, test_y = getMatrices(val_path)

model = learn(train_x, train_y)
predictions = classify(test_x, model)

# Calculate accuracy of predictions
accuracy = accuracy_score(test_y, predictions)
print(f"Accuracy of A4testbed code: {accuracy * 100:.2f}%")


############################ random-forest approach ############################

def load_dataset(file_path):
  train_data = np.loadtxt(file_path, delimiter=',')
  y = train_data[:, 0]
  X = train_data[:, 1:] / 255.
  return X, y

def train_test_random_forest(X_train, y_train, X_test, y_test):
  # Normalize data
  X_train = X_train / 255.0
  X_test = X_test / 255.0
  
  # Train Random Forest
  rf = RandomForestClassifier(n_estimators=100, random_state=42)
  rf.fit(X_train, y_train)
  
  # Predict and evaluate
  y_pred_train = rf.predict(X_train)
  y_pred_test = rf.predict(X_test)
  train_acc = accuracy_score(y_train, y_pred_train)
  test_acc = accuracy_score(y_test, y_pred_test)
  
  # print(f"Training Accuracy for Random Forest: {train_acc * 100:.2f}%")
  # print(f"Test Accuracy for Random Forest: {test_acc * 100:.2f}%")
  return test_acc

# Function to train and test SVM
def train_test_svm(X_train, y_train, X_test, y_test):
  # Normalize data
  X_train = X_train / 255.0
  X_test = X_test / 255.0
  
  # Train SVM
  svm = SVC(kernel='rbf', C=1.0, gamma='scale')
  svm.fit(X_train, y_train)
  
  # Predict and evaluate
  y_pred_train = svm.predict(X_train)
  y_pred_test = svm.predict(X_test)
  train_acc = accuracy_score(y_train, y_pred_train)
  test_acc = accuracy_score(y_test, y_pred_test)
  
  # print(f"Training Accuracy for SVM: {train_acc * 100:.2f}%")
  # print(f"Test Accuracy for SVM: {test_acc * 100:.2f}%")
  return test_acc

def train_test_logistic_regression(X_train, y_train, X_test, y_test):
  # Normalize data
  X_train = X_train / 255.0
  X_test = X_test / 255.0
  
  # Train Logistic Regression
  lr = LogisticRegression(max_iter=1000, solver='lbfgs')
  lr.fit(X_train, y_train)
  
  # Predict and evaluate
  y_pred_train = lr.predict(X_train)
  y_pred_test = lr.predict(X_test)
  train_acc = accuracy_score(y_train, y_pred_train)
  test_acc = accuracy_score(y_test, y_pred_test)
  
  # print(f"Training Accuracy for Logistic Regression: {train_acc * 100:.2f}%")
  # print(f"Test Accuracy for Logistic Regression: {test_acc * 100:.2f}%")
  return test_acc

X_train, y_train = load_dataset(train_path)  # Load your dataset
X_test, y_test = load_dataset(val_path)  # Load your dataset

def train_voting_classifier(X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
  """
  Train a Voting Classifier using Logistic Regression, SVM, and Random Forest.

  Args:
      X_train (np.ndarray): Training features.
      y_train (np.ndarray): Training labels.

  Returns:
      VotingClassifier: Trained Voting Classifier.
  """
  lr_clf = LogisticRegression(max_iter=1000, random_state=42)
  svm_clf = SVC(probability=True, random_state=42)  # Enable probability for soft voting
  rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)

  voting_clf = VotingClassifier(
      estimators=[('lr', lr_clf), ('svm', svm_clf), ('rf', rf_clf)],
      voting='soft'  # Soft voting (average probabilities)
  )
  voting_clf.fit(X_train, y_train)
  return voting_clf

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
  """
  Evaluate a trained model on the test set.

  Args:
      model: Trained classifier.
      X_test (np.ndarray): Test features.
      y_test (np.ndarray): Test labels.
  """
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy:.3f}")

  return accuracy


voting_model = train_voting_classifier(X_train, y_train)
voting_model_test_acc = evaluate_model(voting_model, X_test, y_test)

# Print nice table of results, comparing all the approaches above and their accuracies
print("\n\n")
print("Algorithm\t\t\tAccuracy")
print("---------\t\t\t--------")
print(f"A4testbed code\t\t\t{accuracy * 100:.2f}%")
print(f"Voting Classifier\t\t{voting_model_test_acc * 100:.2f}%")
print(f"Random Forest\t\t\t{train_test_random_forest(X_train, y_train, X_test, y_test) * 100:.2f}%")
print(f"Logistic Regression\t\t{train_test_logistic_regression(X_train, y_train, X_test, y_test) * 100:.2f}%")
print(f"SVM\t\t\t\t{train_test_svm(X_train, y_train, X_test, y_test) * 100:.2f}%")
print("\n\n")
