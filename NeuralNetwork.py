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
import statistics as stats
import matplotlib.pyplot as plt
import onnxruntime

class Dose_DataSet(torch.utils.data.DataLoader):
    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            if scale_data:
                scaler = StandardScaler()
                scale_params_X = scaler.fit(X)
                X = scaler.fit_transform(X)
                
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
            self.scale_params = scale_params_X
            print("Initialized dataset.")
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
def DataScaler_Training(path):
    #take the compiled input array and save the mean and the standard deviation for each feature, and scale data so that it has mean 0 and variance 1. 
    #do this before feature channels have been reshaped to be independent features (so there are 36 features = 12 x 3)
    files = os.listdir(path) 
    num_training_files = int(len(files))
    with open (os.path.join(os.getcwd(), "Saved_Values", "new_none_values.txt"), "rb") as fp:
        new_none_values = pickle.load(fp)
    training_files = files[0:num_training_files]
    X = np.zeros((num_training_files, 12 , 3))

    for idx, file in enumerate(training_files):
        file_path = os.path.join(path, file)
        with open(file_path, "rb") as fp:
            x,y = pickle.load(fp)
            X[idx,:,:] = x

    means = np.zeros(X.shape[1:])
    stds = np.zeros(X.shape[1:])
    if len(X.shape) == 3:
        f_range = range(X.shape[1])
        c_range = range(X.shape[2])  
        for f in f_range:
            for c in c_range:
                vals = []
                for s in range(X.shape[0]):
                    if X[s,f,c] == 1000:
                        val = new_none_values[f,c]
                    else:
                        val = X[s,f,c]    
                    vals.append(val)
                std = stats.pstdev(vals)
                mean = stats.mean(vals) 
                means[f,c] = mean
                stds[f,c] = std   
                # for s in range(X.shape[0]):
                #     X_scaled[s,f,c] = (X[s,f,c] - mean)/std     
    else:
        raise Exception("Could not understand input dimensions.")
  

    with open (os.path.join(os.getcwd(), "Saved_Values", "ScaleParams.txt"), "wb") as fp:
        pickle.dump([means,stds], fp)
  

def Scale_Data(X):
    with open (os.path.join(os.getcwd(), "Saved_Values", "ScaleParams.txt"), "rb") as fp:
        means, stds  = pickle.load(fp)
    with open (os.path.join(os.getcwd(), "Saved_Values", "new_none_values.txt"), "rb") as fp:
        new_none_values = pickle.load(fp)    
    f_range = range(X.shape[0])
    c_range = range(X.shape[1])
    for f in f_range:
        for c in c_range:
            if X[f,c] == 1000:
                X[f,c] = new_none_values
            X[f,c] = (X[f,c] - means[f,c]) / stds[f,c]  
    return X             




def Predict(input):
    input = Scale_Data(input)
    input = np.reshape(input, (1,input.size))
    input = torch.from_numpy(input).float()
    mlp = MLP()  
    mlp.load_state_dict(torch.load(os.path.join(os.getcwd(), "Saved_Values", "Model")))  
    prediction = mlp(input)
    prediction = prediction.detach().numpy()         
    return prediction
    print("")          

def Train(path):
    #first need to split data into training/validation set. Will do a 85/15 split
    DataScaler_Training(path)
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
            X_training_stack[idx,:,:] = X
            y_training_stack[idx, :,:] = y
            
    X_validation_stack = np.zeros((num_validation_files, 12 , 3))
    y_validation_stack = np.zeros((num_validation_files, 9 , 5))
    for idx, file in enumerate(validation_files):
        file_path = os.path.join(path, file)
        with open(file_path, "rb") as fp:
            X,y = pickle.load(fp)

            X_validation_stack[idx,:] = X
            y_validation_stack[idx, :] = y  
            
    X_training_stack, y_training_stack , X_validation_stack, y_validation_stack =  Statistics.Scale_NonExisting_Features(X_training_stack, y_training_stack , X_validation_stack, y_validation_stack)       
    prediction = Predict(X_validation_stack[0,:,:])  
    real = y_validation_stack[0,:,:]
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
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_history = []
    for epoch in range(0,100):
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
    #save the loss history
    with open (os.path.join(os.getcwd(), "Saved_Values", "LossHistory.txt"), "wb") as fp:
        pickle.dump(loss_history, fp)
    #save the model     
    torch.save(mlp.state_dict(), os.path.join(os.getcwd(), "Saved_Values", "Model.onnx"))    
    
    x_range = np.linspace(1, 3001, 3000)
    y_range = loss_history
    plt.plot(x_range, y_range, label = "Loss")
    plt.xlabel('Epoch')
    plt.ylabel("Batch Loss")
    plt.title("Lostt History")
    print("Loss History During Training")  
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(os.getcwd(), "Saved_Values", "LossHistoryPlot.jpg"))


    """Takes a PyTorch model and converts it to an Open Neural Network Exchange 
       (ONNX) model for operability with other programming languages. The new 
       model is saved into the Models folder.

    Args:
        organ (str): the name of the organ for which a PyTorch model is to 
            be converted to an ONNX model
        modelType (str): specifies whether a UNet or MultiResUnet model is 
            to be converted

    """



    #Now need a dummy image to predict with to save the weights
    x = np.zeros((512,512))
    x = torch.from_numpy(x)
    x = torch.reshape(x, (1,1,512,512)).float()
    torch.onnx.export(mlp,x,os.path.join(os.getcwd(), str("dose_model.onnx")), export_params=True, opset_version=10)
    try:
        session = onnxruntime.InferenceSession(os.getcwd(), str("dose_model.onnx"))
    except (TypeError, RuntimeError) as e:
        raise e




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






