import torch 
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

class Dose_DataSet(torch.utils.data.DataLoader):
    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i] 

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )              
    def forward(self, x):

        return self.layers(x)     

if __name__ == "__main__":
    torch.manual_seed(42)
    X, y = load_boston(return_X_y=True)    
    dataset = Dose_DataSet(X,y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    mlp = MLP()
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    for epoch in range(0,50):
        print(f'Starting epoch {epoch+1}')
        current_loss = 0

        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            optimizer.zero_grad() #zero gradients

            outputs = mlp(inputs)
            
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            if i % 10 == 0:
                print('loss after mini-batch %5d: %.3f' % 
                      (i + 1, current_loss / 500))






