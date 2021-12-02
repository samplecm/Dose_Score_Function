import torch 
import torch.nn as nn
import numpy as np 
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import os
import pickle 
import Statistics

class Dose_DataSet(torch.utils.data.DataLoader):
    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            if scale_data:
                scale_params_X = StandardScaler().fit(X)
                X = StandardScaler().fit_transform(X)
                
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
            self.scale_params = scale_params_X

    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i] 

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(36, 32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 45)
        )              
    def forward(self, x):

        return self.layers(x)     

def Train(path):
    #first need to split data into training/validation set. Will do a 85/15 split
    files = os.listdir(path) 
    num_training_files = int(len(files) * 0.85)
    random.shuffle(files)
    training_files = files[0:num_training_files]
    validation_files = files[num_training_files:]
    num_validation_files = len(validation_files)
    X_training_stack = np.zeros((num_training_files, 12 , 3))
    y_training_stack = np.zeros((num_training_files, 9 , 5))
    for idx, file in enumerate(training_files):
        file_path = os.path.join(path, file)
        with open(file_path, "rb") as fp:
            X,y = pickle.load(fp)
            X_training_stack[idx,:] = X
            y_training_stack[idx, :] = y
            
    X_validation_stack = np.zeros((num_validation_files, 12 , 3))
    y_validation_stack = np.zeros((num_validation_files, 9 , 5))
    for idx, file in enumerate(validation_files):
        file_path = os.path.join(path, file)
        with open(file_path, "rb") as fp:
            X,y = pickle.load(fp)

            X_validation_stack[idx,:] = X
            y_validation_stack[idx, :] = y  
    X_training_stack, y_training_stack , X_validation_stack, y_validation_stack =  Statistics.Scale_NonExisting_Features(X_training_stack, y_training_stack , X_validation_stack, y_validation_stack)       
      
    #reshape arrays 
    X_training_stack = np.reshape(X_training_stack, (X_training_stack.shape[0], X_training_stack.shape[1]*X_training_stack.shape[2]))
    y_training_stack = np.reshape(y_training_stack, (y_training_stack.shape[0], y_training_stack.shape[1]*y_training_stack.shape[2]))
    X_validation_stack = np.reshape(X_validation_stack, (X_validation_stack.shape[0], X_validation_stack.shape[1]*X_validation_stack.shape[2]))
    y_validation_stack = np.reshape(y_validation_stack, (y_validation_stack.shape[0], y_validation_stack.shape[1]*y_validation_stack.shape[2]))
    

    #Now create a dataloader object
    train_set = Dose_DataSet(X_training_stack,y_training_stack)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    validation_set = Dose_DataSet(X_validation_stack,y_validation_stack)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=True, num_workers=1)
    mlp = MLP()
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    loss_history = []
    for epoch in range(0,1000):
        print(f'Starting epoch {epoch+1}')
        
        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 45))

            outputs = mlp(inputs)
            
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() #zero gradients

        total_loss = 0
        for i, data in enumerate(validation_loader):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 45))

            outputs = mlp(inputs)
            
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() #zero gradients
        loss_history.append(round(total_loss / (i+1),2))    
        print(f'Validation Loss: {round(total_loss / (i+1),2)}')        

    
    print("")  






    print("Completed Training")      

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






