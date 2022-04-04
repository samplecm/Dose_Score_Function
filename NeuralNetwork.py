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
import statistics 

try:
    patients_path = os.path.join(os.getcwd(), "Patients")
    processed_path = os.path.join(os.getcwd(), "Processed_Patients")
    training_path = os.path.join(os.getcwd(), "Training_Data")
    statistics_path = os.path.join(os.getcwd(), "Statistics")
except: 
    patients_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Programs/Dose_Score_Function/20211110_Caleb_SGFX"    
    processed_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Programs/Dose_Score_Function/Processed_Patients"
    training_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Programs/Dose_Score_Function/Processed_Patients/Training_Data"

organs = [
        "brainstem",
        "larynx",
        "mandible", 
        "oral_cavity",
        "parotid_left",
        "parotid_right", 
        "spinal_cord", 
        "submandibular_right",
        "submandibular_left", 
    ]

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

def Get_OAR_Distances():
    #adds another list attribute to each patient (oar_dists) which contains the radial distance to each other whole oar (1111 if not present)
    processed_patients = os.listdir(processed_path)
    processed_patients.sort()
    all_dists = [] 
    for patient_path in processed_patients:
        with open(os.path.join(processed_path, patient_path), "rb") as fp:
            patient = pickle.load(fp)
            for oar in organs:
                oar_obj = getattr(patient, oar)
                if oar_obj is None:
                    continue
                oar_dist_subsegs = []
                for subseg_centre in oar_obj.centre_point_subsegs:
                    oar_dists = []

                    for oar_2 in organs:
                        if oar_2 == oar:
                            continue
                        
                        oar_obj2 = getattr(patient, oar_2)

                        if oar_obj2 is None:
                            oar_dists.append(1111)
                            continue

                        oar_pos_2 = oar_obj2.centre_point
                        oar_dists.append(np.sqrt((subseg_centre[1]-oar_pos_2[0])**2 + (subseg_centre[1]-oar_pos_2[1])**2 + (subseg_centre[2]-oar_pos_2[2])**2))
                        all_dists.append(np.sqrt((subseg_centre[1]-oar_pos_2[0])**2 + (subseg_centre[1]-oar_pos_2[1])**2 + (subseg_centre[2]-oar_pos_2[2])**2))

                    oar_dist_subsegs.append(oar_dists)
                oar_obj.oar_distances_subsegs = oar_dist_subsegs 

        with open(os.path.join(processed_path, patient_path), "wb") as fp:
            pickle.dump(patient, fp)     
        print(f"Finished getting oar distances for {patient_path}")    
    oar_dist_stats = [statistics.mean(all_dists), statistics.stdev(all_dists)]    
    with open(os.path.join(statistics_path, "oar_dist_stats"), "wb") as fp:
        pickle.dump(oar_dist_stats, fp)      

        






def Get_Distance_Stats():
    #returns and saves to stats directory the mean and std for min distance, max distance
    processed_patients = os.listdir(processed_path)
    min_dists = []
    # max_dists = []
    

    for patient_path in processed_patients:
        with open(os.path.join(processed_path, patient_path), "rb") as fp:
            patient = pickle.load(fp)
            for oar in organs:
                oar_obj = getattr(patient, oar)
                if oar_obj is None:
                    continue
                all_subseg_data = oar_obj.spatial_data_subsegs
                for subseg_data in all_subseg_data:
                    for ptv_data in subseg_data:
                        min_data = ptv_data[2]
                        #max_data = ptv_data[3]
                        for i in range(len(min_data)):
                            if min_data[i][0] != 1111:
                                min_dists.append(min_data[i][0])
                            # max_dists.append(max_data[i][0])
        print(f"Finished getting ptv distance data for {patient_path}")                       

    min_stats = [statistics.mean(min_dists), statistics.stdev(min_dists)]    
    with open(os.path.join(statistics_path, "min_stats"), "wb") as fp:
        pickle.dump(min_stats, fp)

    # max_stats = [statistics.mean(max_dists), statistics.stdev(max_dists)]    
    # with open(os.path.join(statistics_path, "max_stats"), "wb") as fp:
    #     pickle.dump(max_stats, fp)    

def Get_Training_Arrays():
    processed_patients = os.listdir(processed_path)
    #get stats for normalizing data
    with open(os.path.join(statistics_path, "oar_dist_stats"), "rb") as fp:
        oar_distance_stats = pickle.load(fp)  
    with open(os.path.join(statistics_path, "min_stats"), "rb") as fp:
        min_distance_stats = pickle.load(fp)   


    for patient_path in processed_patients:
        with open(os.path.join(processed_path, patient_path), "rb") as fp:
            patient = pickle.load(fp)
            for oar in organs:
                oar_obj = getattr(patient, oar)
                if oar_obj is None:
                    continue
                all_subseg_data = oar_obj.spatial_data_subsegs
                
                for idx in range(len(all_subseg_data)):
                    if not os.path.exists(os.path.join(os.getcwd(), "Training_Data", oar, str(idx))):
                        os.mkdir(os.path.join(os.getcwd(), "Training_Data", oar, str(idx)))
                    #if more than 4 ptv types, combine the bottom  prescription (minimum distance of both, with max ptv num)    
                    training_array = [] 
                    #first add oar distances 

                    for distance in oar_obj.oar_distances_subsegs[idx]:
                        if distance == 1111:
                            z = 10
                        else:    
                            z = (distance - oar_distance_stats[0])/oar_distance_stats[1] #statistical z value
                        training_array.append(z)

                    for ptv_idx in range(min(len(all_subseg_data[idx]), 2)):
                        training_array.append(all_subseg_data[idx][ptv_idx][0])
                        training_array.append(all_subseg_data[idx][ptv_idx][1])
                        for point in all_subseg_data[idx][ptv_idx][2]:
                            if point[0] ==1111:
                                training_array.append(10)
                                training_array.append(0)
                                training_array.append(0)
                            else:    
                                training_array.append((point[0]-min_distance_stats[0])/min_distance_stats[1])
                                training_array.append(point[1])
                                training_array.append(point[2])

                    if len(all_subseg_data[idx]) > 3:
                        ptv_types = []
                        overlap_frac = 0
                        min_points =[[10000, 0,0]]*18
                        for ptv_idx in range(len(all_subseg_data[idx])-2): 
                            ptv_types.append(all_subseg_data[idx][ptv_idx][0])
                            if all_subseg_data[idx][ptv_idx][1] > overlap_frac:
                                overlap_frac = all_subseg_data[idx][ptv_idx][1]
                            for p, point in enumerate(all_subseg_data[idx][ptv_idx][2]):    
                                if point[0] < min_points[p][0]:
                                    min_points[p] = point   
                        bottom_ptv = statistics.mean(ptv_types)              
                        patient.num_ptv_types = len(all_subseg_data[idx])

                        training_array.append(bottom_ptv)
                        training_array.append(overlap_frac)
                        for point in min_points:
                            if point[0] ==1111:
                                training_array.append(10)
                                training_array.append(0)
                                training_array.append(0)
                            else:    
                                training_array.append((point[0]-min_distance_stats[0])/min_distance_stats[1])
                                training_array.append(point[1])
                                training_array.append(point[2])

                    if len(all_subseg_data[idx]) == 3:    
                        training_array.append(all_subseg_data[idx][2][0])
                        training_array.append(all_subseg_data[idx][2][1])
                        for point in all_subseg_data[idx][2][2]:
                            if point[0] ==1111:
                                training_array.append(10)
                                training_array.append(0)
                                training_array.append(0)
                            else:    
                                training_array.append((point[0]-min_distance_stats[0])/min_distance_stats[1])
                                training_array.append(point[1])
                                training_array.append(point[2])    
                    converted_array = np.linspace(0,0,176)
                    for v, val in enumerate(training_array):
                        converted_array[v] = val
                                
                    with open(os.path.join(os.getcwd(), "Training_Data", oar, str(idx), str(patient_path+"_data.txt")), "wb") as fp:
                        pickle.dump(converted_array, fp)

                    



                        

        print(f"Finished getting training array for {patient_path}")
            



                

            

    

def PreprocessData():
    #Get_OAR_Distances()
    #Get_Distance_Stats()
    #Get_Training_Arrays()
    Cross_Validate()


    # processed_patients = os.listdir(processed_path)
    # for patient_path in processed_patients:
    #    with open(os.path.join(processed_path, patient_path), "rb") as fp:
    #        patient = pickle.load(fp)



def TrainModels():
    PreprocessData()




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






