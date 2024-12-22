import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from torch import nn, optim
from typing import Tuple

def learn(X, y) -> nn.Module:
  """
  Train a neural network model on the provided dataset.

  Args:
      X (torch.Tensor): Input feature matrix (n x d).
      y (torch.Tensor): Target labels (n x 1).

  Returns:
      nn.Module: Trained PyTorch neural network model.
  """

  class BasicNeuralNetwork(nn.Module):
    def __init__(self):
      super(BasicNeuralNetwork, self).__init__()
      self.fc1 = nn.Linear(2352, 512)
      self.bn1 = nn.BatchNorm1d(512)
      self.fc2 = nn.Linear(512, 256)
      self.bn2 = nn.BatchNorm1d(256)
      self.fc3 = nn.Linear(256, 128)
      self.bn3 = nn.BatchNorm1d(128)
      self.fc4 = nn.Linear(128, 10)
      self.dropout = nn.Dropout(0.4)

    def forward(self, x):
      x = F.relu(self.bn1(self.fc1(x)))
      x = self.dropout(x)
      x = F.relu(self.bn2(self.fc2(x)))
      x = self.dropout(x)
      x = F.relu(self.bn3(self.fc3(x)))
      x = self.dropout(x)
      x = self.fc4(x)
      return F.log_softmax(x, dim=1)

  # Convert the input matrices to PyTorch tensors
  X = torch.Tensor(X)
  y = torch.LongTensor(y)

  # Prepare dataset
  dataset = TensorDataset(X, y)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  # Initialize model, optimizer, and scheduler
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = BasicNeuralNetwork().to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

  # Training loop
  n_epochs = 50
  m = len(dataloader.dataset)
  for _ in range(n_epochs):
    model.train()
    total_loss = 0
    correct = 0
    for data, target in dataloader:
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

    test_average_loss = total_loss / m
    accuracy = correct / m

    print(f'Test average loss: {test_average_loss:.4f}, accuracy: {accuracy:.3f}')
    scheduler.step()  # Adjust learning rate

  return model

def classify(Xtest, model) -> torch.Tensor:
  """
  Classify the test dataset using the trained model.

  Args:
      Xtest (torch.Tensor): Input feature matrix (m x d).
      model (nn.Module): Trained PyTorch neural network model.

  Returns:
      torch.Tensor: Predicted labels (m x 1).
  """

  # Convert input matrix to PyTorch tensor
  Xtest = torch.Tensor(Xtest)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.eval()
  Xtest = Xtest.to(device)
  with torch.no_grad():
    output = model(Xtest)
    predictions = output.argmax(dim=1)
  return predictions