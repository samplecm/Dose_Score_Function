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
import csv
import DVH_Fitting
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
        "oral_cavity",
        "parotid_left",
        "parotid_right", 
        "spinal_cord", 
        "submandibular_right",
        "submandibular_left", 
    ]

 
# def DataScaler_Training(path):
#     #take the compiled input array and save the mean and the standard deviation for each feature, and scale data so that it has mean 0 and variance 1. 
#     #do this before feature channels have been reshaped to be independent features (so there are 36 features = 12 x 3)
#     files = os.listdir(path) 
#     num_training_files = int(len(files))
#     with open (os.path.join(os.getcwd(), "Saved_Values", "new_none_values.txt"), "rb") as fp:
#         new_none_values = pickle.load(fp)
#     training_files = files[0:num_training_files]
#     X = np.zeros((num_training_files, 12 , 3))

#     for idx, file in enumerate(training_files):
#         file_path = os.path.join(path, file)
#         with open(file_path, "rb") as fp:
#             x,y = pickle.load(fp)
#             X[idx,:,:] = x

#     means = np.zeros(X.shape[1:])
#     stds = np.zeros(X.shape[1:])
#     if len(X.shape) == 3:
#         f_range = range(X.shape[1])
#         c_range = range(X.shape[2])  
#         for f in f_range:
#             for c in c_range:
#                 vals = []
#                 for s in range(X.shape[0]):
#                     if X[s,f,c] == 1000:
#                         val = new_none_values[f,c]
#                     else:
#                         val = X[s,f,c]    
#                     vals.append(val)
#                 std = stats.pstdev(vals)
#                 mean = stats.mean(vals) 
#                 means[f,c] = mean
#                 stds[f,c] = std   
#                 # for s in range(X.shape[0]):
#                 #     X_scaled[s,f,c] = (X[s,f,c] - mean)/std     
#     else:
#         raise Exception("Could not understand input dimensions.")
  

#     with open (os.path.join(os.getcwd(), "Saved_Values", "ScaleParams.txt"), "wb") as fp:
#         pickle.dump([means,stds], fp)
  

# def Scale_Data(X):
#     with open (os.path.join(os.getcwd(), "Saved_Values", "ScaleParams.txt"), "rb") as fp:
#         means, stds  = pickle.load(fp)
#     with open (os.path.join(os.getcwd(), "Saved_Values", "new_none_values.txt"), "rb") as fp:
#         new_none_values = pickle.load(fp)    
#     f_range = range(X.shape[0])
#     c_range = range(X.shape[1])
#     for f in f_range:
#         for c in c_range:
#             if X[f,c] == 1000:
#                 X[f,c] = new_none_values
#             X[f,c] = (X[f,c] - means[f,c]) / stds[f,c]  
#     return X             


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

def Get_Training_Data(organs):
    processed_patients = os.listdir(processed_path)
    with open(os.path.join(processed_path, processed_patients[0]), "rb") as fp:
        example_patient = pickle.load(fp)
    for oar in organs:
        oar_obj_ex = getattr(example_patient, oar)
        oar_dist_subsegs_ex = oar_obj_ex.oar_distances_subsegs
        for s in range(len(oar_dist_subsegs_ex)):
            print("")
    print("Finished processing training data")
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
                        training_array.append(all_subseg_data[idx][ptv_idx][0])   #ptv type
                        training_array.append(all_subseg_data[idx][ptv_idx][1])   #overlap frac
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
                    #also need the dvh parameters (y)
                    y = np.array(oar_obj.dvh_params_subsegs[idx][0:2])    
                                
                    with open(os.path.join(os.getcwd(), "Training_Data", oar, str(idx), str(patient_path+"_data.txt")), "wb") as fp:
                        pickle.dump([converted_array, y], fp)

                    



                        

        print(f"Finished getting training array for {patient_path}")

def Train_Pruned_Model(dir, oar_dir, subseg_num, organ):
    cross_val_dir = os.path.join(oar_dir, "cross_validation", str("subseg_" + subseg_num))    
    data_files = os.listdir(dir)
    with open(os.path.join(cross_val_dir, "test_files.txt"), "rb") as fp:
        test_files = pickle.load(fp)
    with open (os.path.join(cross_val_dir, "Percent_Diffs.txt"), "rb") as fp:
        percent_diffs = pickle.load(fp)
    percent_diff_names = []
    percent_diff_vals = []
    for item in percent_diffs:
        percent_diff_names.append(item[0])
        percent_diff_vals.append(item[1][0])
    #now omit files that predicted dose centres greater than 5% lower than actual
    omit_list = []
    for i, item in enumerate(percent_diffs):
        if item < -0.05:    
            omit_list.append(percent_diff_names[i])                        
    train_files = [file for file in data_files if file not in test_files and file not in omit_list]  
    test_stats = Get_Pruned_Test_Stats(train_files, test_files, dir) 
    with open (os.path.join(cross_val_dir, "test_stats_pruned.txt"), "wb") as fp:
        pickle.dump(test_stats, fp)   
    #now will transfer training patients in top 10% for most negative absolute diff of first parameter prediction from real






def Cross_Validate(dir, oar_dir, subseg_num, organ, fold=10):
    if not os.path.exists(os.path.join(oar_dir, "cross_validation")):
        os.mkdir(os.path.join(oar_dir, "cross_validation"))
    

    data_files = os.listdir(dir)
    random.shuffle(data_files)
    if len(data_files) < 100:
        return

    cross_val_dir = os.path.join(oar_dir, "cross_validation", str("subseg_" + subseg_num))    
    if not os.path.exists(cross_val_dir):
        os.mkdir(cross_val_dir)    
    for i in range(fold):
        fold_dir = os.path.join(cross_val_dir, str(i))
        if not os.path.exists(fold_dir):    #make folder for each cross val fold
            os.mkdir(os.path.join(cross_val_dir, str(i)))

    test_size = int(len(data_files)*0.05)
    test_files = data_files[0:test_size]

    cross_val_files = data_files[test_size:]
    val_size = int(0.1*(len(cross_val_files)))
    percent_diffs_all = []
    for i in range(fold):
        print(f"starting fold {i+1}")
        fold_dir = os.path.join(cross_val_dir, str(i))
        val_files = cross_val_files[i*val_size:(i+1)*val_size]
        train_files = [file for file in cross_val_files if file not in val_files]
        train_stack, val_stack, train_stack_y, val_stack_y = Get_Training_Stacks(train_files, val_files, dir)

        train_set = DataSet(train_stack,train_stack_y)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False, num_workers=1)
        validation_set = DataSet(val_stack, val_stack_y)
        val_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=1) 

        model, loss_history, percent_diffs = Train_Model(train_loader, val_loader)
        for j, item in enumerate(percent_diffs):
            percent_diffs_all.append([val_files[j], item])

        with open (os.path.join(fold_dir, "LossHistory.txt"), "wb") as fp:
            pickle.dump(loss_history, fp)
        #save the model     
        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))   
        print(f"Finished Cross Validation training for fold {i+1} of subsegment {subseg_num} of {organ}")  

    with open (os.path.join(cross_val_dir, "Percent_Diffs.txt"), "wb") as fp:
        pickle.dump(percent_diffs_all, fp)        
    with open (os.path.join(cross_val_dir, "test_files.txt"), "wb") as fp:
        pickle.dump(test_files, fp)   

    test_stats, model = Get_Test_Stats(test_files, dir)  
    with open (os.path.join(cross_val_dir, "test_stats.txt"), "wb") as fp:
        pickle.dump(test_stats, fp)     
    torch.save(model.state_dict(), os.path.join(cross_val_dir, "test_model.pt"))     
    print(f"Finished Cross Validation training")   



def Get_Training_Stacks(train_files, val_files, path):
    train_stack = np.zeros((len(train_files), 176))
    val_stack = np.zeros((len(val_files), 176))
    train_stack_y = np.zeros((len(train_files), 2))
    val_stack_y = np.zeros((len(val_files), 2))

    train_idx = 0
    val_idx = 0
    for file in os.listdir(path):
        if file in train_files:
            data = pickle.load(open(os.path.join(path, file), "rb"))
            train_stack[train_idx,:] = data[0]
            train_stack_y[train_idx,:] = data[1]
            train_idx += 1
        if file in val_files:     
            data = pickle.load(open(os.path.join(path, file), "rb"))
            val_stack[val_idx,:] = data[0]
            val_stack_y[val_idx,:] = data[1]
            val_idx += 1
    return train_stack, val_stack, train_stack_y, val_stack_y

def Train_Test_Model(test_loader):
    mlp = MLP()
    loss_function = nn.MSELoss()#nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-5)
    for epoch in range(0,800):
        print(f'Starting epoch {epoch+1}')
        
        for i, data in enumerate(test_loader):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()

            outputs = mlp(inputs)
            
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() #zero gradients
    return mlp 
        

def Train_Model(train_loader, val_loader):
    mlp = MLP()
    loss_function = nn.MSELoss()#nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-5)
    loss_history = []
    for epoch in range(0,1200):
        print(f'Starting epoch {epoch+1}')
        
        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()

            outputs = mlp(inputs)
            
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() #zero gradients

        total_loss = 0
        for i, data in enumerate(val_loader):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()

            outputs = mlp(inputs)
            
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() #zero gradients

        loss_history.append(round(total_loss / (i+1),2))    
        print(f'Validation Loss: {round(total_loss / (i+1),7)}')  
    percent_diffs = []
    for i, data in enumerate(val_loader):    #get % diffs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()

        outputs = mlp(inputs)
        targets = targets.detach().numpy()
        outputs = outputs.detach().numpy()
        percent_diff = [(outputs[0][0]-targets[0][0]), (outputs[0][1]-targets[0][1])]    #used for model pruning
        percent_diffs.append(percent_diff)

 

    return mlp, loss_history, percent_diffs         
    #save the loss history
    # with open (os.path.join(os.getcwd(), "Saved_Values", "LossHistory.txt"), "wb") as fp:
    #     pickle.dump(loss_history, fp)
    # #save the model     
    # torch.save(mlp.state_dict(), os.path.join(os.getcwd(), "Saved_Values", "Model.onnx"))   

def Get_Pruned_Test_Stats(train_files, test_files, dir):
    files = os.listdir(dir)
    stack = np.zeros((len(test_files), 176))
    stack_y = np.zeros((len(test_files), 2))
    train_stack = np.zeros((len(train_files), 176))
    train_stack_y = np.zeros((len(train_files), 2))

    idx = 0
    train_idx = 0
    for file in files:
        if file in test_files:
            data = pickle.load(open(os.path.join(dir, file), "rb"))
            stack[idx,:] = data[0]
            stack_y[idx,:] = data[1]
            idx += 1
        elif file in train_files:  
            data = pickle.load(open(os.path.join(dir, file), "rb"))
            train_stack[train_idx,:] = data[0]
            train_stack_y[train_idx,:] = data[1]
            train_idx += 1  

    test_set = DataSet(stack,stack_y)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)  
    train_set = DataSet(train_stack,train_stack_y)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False, num_workers=1)      
    model = Train_Test_Model(train_loader)
    actual_params = []    
    params = []
    for i, data in enumerate(test_loader):    #get % diffs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()

        outputs = model(inputs)
        targets = targets.detach().numpy()
        outputs = outputs.detach().numpy()
        actual_params.append([targets[0][0], targets[0][1]])
        params.append([outputs[0][0], outputs[0][1]])    
    return [params, actual_params]   

def Get_Test_Stats(test_files, dir):
    files = os.listdir(dir)
    stack = np.zeros((len(test_files), 176))
    stack_y = np.zeros((len(test_files), 2))
    train_stack = np.zeros((len(files)-len(test_files), 176))
    train_stack_y = np.zeros((len(files)-len(test_files), 2))

    idx = 0
    test_idx = 0
    for file in files:
        if file in test_files:
            data = pickle.load(open(os.path.join(dir, file), "rb"))
            stack[idx,:] = data[0]
            stack_y[idx,:] = data[1]
            idx += 1
        else:  
            data = pickle.load(open(os.path.join(dir, file), "rb"))
            train_stack[test_idx,:] = data[0]
            train_stack_y[test_idx,:] = data[1]
            test_idx += 1  

    test_set = DataSet(stack,stack_y)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)  
    train_set = DataSet(train_stack,train_stack_y)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False, num_workers=1)      
    model = Train_Test_Model(train_loader)
    actual_params = []    
    params = []
    for i, data in enumerate(test_loader):    #get % diffs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()

        outputs = model(inputs)
        targets = targets.detach().numpy()
        outputs = outputs.detach().numpy()
        actual_params.append([targets[0][0], targets[0][1]])
        params.append([outputs[0][0], outputs[0][1]])    
    return [params, actual_params], model     


class DataSet(torch.utils.data.DataLoader):
    def __init__(self, X, y):
        if not torch.is_tensor(X) and not torch.is_tensor(y):               
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
            nn.Linear(176, 200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )              
    def forward(self, x):

        return self.layers(x)             

    
    
def Plot_Percent_Diffs():
    print("")

def Get_Test_Stats_CSV():
    for organ in organs:
        with open(os.path.join(statistics_path, str(organ + '_test_stats.csv')), 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            oar_dir = os.path.join(training_path, organ)
            subseg_dirs = os.listdir(oar_dir)
            subseg_dirs.sort() 
            filewriter.writerow(['', 'P1 avg','P1 std', 'P1 percent diff', 'P1 percent diff std', 'P2 avg', 'P2 std', 'P2 percent diff', 'P2 percent diff std'])
            for subseg in subseg_dirs:
                if subseg == "cross_validation":
                    continue
                cross_val_dir = os.path.join(oar_dir, "cross_validation", str("subseg_" + subseg)) 
                with open (os.path.join(cross_val_dir, "test_stats.txt"), "rb") as fp:
                    data = pickle.load(fp)
                param_1 = []
                actual_param_1 = []
                param_2 = []
                actual_param_2 = []
                param_diffs_1 = []
                param_diffs_2 = []
                for idx in range(len(data[0])):
                    param_1.append(data[0][idx][0])
                    param_2.append(data[0][idx][1])
                    actual_param_1.append(data[1][idx][0])
                    actual_param_2.append(data[1][idx][1])
                    param_diffs_1.append(param_1[-1]-actual_param_1[-1])
                    param_diffs_2.append(param_2[-1]-actual_param_2[-1])
                p1_avg = str(statistics.mean(param_1))
                p1_std = str(np.std(param_1))
                p2_avg = str(statistics.mean(param_2))
                p2_std = str(np.std(param_2))
                actual_1_avg = str(statistics.mean(actual_param_1))
                actual_1_std = str(np.std(actual_param_1))
                actual_2_avg = str(statistics.mean(actual_param_2))
                actual_2_std = str(np.std(actual_param_2))
                diff_1_avg = str(statistics.mean(param_diffs_1))
                diff_1_std = str(np.std(param_diffs_1))
                diff_2_avg = str(statistics.mean(param_diffs_2))
                diff_2_std = str(np.std(param_diffs_2))
                filewriter.writerow([subseg, p1_avg, p1_std, p2_avg, p2_std, actual_1_avg, actual_1_std, actual_2_avg, actual_2_std, diff_1_avg, diff_1_std, diff_2_avg, diff_2_std])    

def Get_Pruned_Test_Stats_CSV():
    for organ in organs:
        with open(os.path.join(statistics_path, str(organ + '_test_stats.csv')), 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            oar_dir = os.path.join(training_path, organ)
            subseg_dirs = os.listdir(oar_dir)
            subseg_dirs.sort() 
            filewriter.writerow(['', 'P1 avg','P1 std', 'P2 avg', 'P2 std', 'P1 Actual avg', 'P1 Actual std', 'P2 Actual avg', 'P2 Actual std', 'P1 diff avg', 'P1 diff std',  'P2 diff avg', 'P2 diff std'])
            for subseg in subseg_dirs:
                if subseg == "cross_validation":
                    continue
                cross_val_dir = os.path.join(oar_dir, "cross_validation", str("subseg_" + subseg)) 
                with open (os.path.join(cross_val_dir, "test_stats_pruned.txt"), "rb") as fp:
                    data = pickle.load(fp)
                param_1 = []
                diff_1 = []
                param_2 = []
                diff_2 = []
                for idx in range(len(data[0])):
                    param_1.append(data[0][idx][0])
                    param_2.append(data[0][idx][1])
                    diff_1.append(data[1][idx][0])
                    diff_2.append(data[1][idx][1])
                p1_avg = str(statistics.mean(param_1))
                p1_std = str(np.std(param_1))
                p2_avg = str(statistics.mean(param_2))
                p2_std = str(np.std(param_2))
                diff_1_avg = str(statistics.mean(diff_1))
                diff_1_std = str(np.std(diff_1))
                diff_2_avg = str(statistics.mean(diff_2))
                diff_2_std = str(np.std(diff_2))
                filewriter.writerow([subseg, p1_avg, p1_std, p2_avg, p2_std, diff_1_avg, diff_1_std, diff_2_avg, diff_2_std])   


def TrainModels():
    Get_Training_Data()
    for oar in organs:
        oar_dir = os.path.join(training_path, oar)
        subseg_dirs = os.listdir(oar_dir)
        subseg_dirs.sort()
        for subseg_num, dir in enumerate(subseg_dirs):
            if dir == "cross_validation":
                continue
            subseg_dir = os.path.join(oar_dir, dir)
            print(f"Starting cross validation for {dir}th subsegment of {oar}")
            Cross_Validate(subseg_dir, oar_dir, str(dir), oar)
            # print(f"Starting pruned model training for {dir}th subsegment of {oar}")
            # Train_Pruned_Model(subseg_dir, oar_dir, str(dir), oar)
    Get_Test_Stats_CSV()
    #Get_Pruned_Test_Stats_CSV()

    # processed_patients = os.listdir(processed_path)
    # for patient_path in processed_patients:
    #    with open(os.path.join(processed_path, patient_path), "rb") as fp:
    #        patient = pickle.load(fp)
def Get_DVH_Plot(organ):
    oar_dir = os.path.join(training_path, organ)
    # subseg_dirs = os.listdir(oar_dir)
    # subseg_dirs.sort(key=int)
    
    cross_val_dir = model_path = os.path.join(training_path, organ, "cross_validation")
    subseg_dirs = os.listdir(cross_val_dir)
    subseg_dirs.sort()
    test_files = pickle.load(open(os.path.join(cross_val_dir,subseg_dirs[0], "test_files.txt"),"rb"))
    #model_path = os.path.join(cross_val_dir, subseg_dirs[0],"test_model.pt")
    model_path = os.path.join(cross_val_dir, subseg_dirs[1], str(0), "model.pt")

    model = MLP()
    model.load_state_dict(torch.load(model_path))  
    for test_file in test_files:
        Plot_DVH(model, oar_dir, test_file)

def Plot_DVH(model, organ_dir, file_name):
    #this function computes predicted and actual cumulative dvhs from all the differential dvhs of subsegments. It plots them all together     

    subseg_dirs = os.listdir(organ_dir)
    subseg_dirs.sort()
    volumes_predicted = []
    volumes = []
    for dir in subseg_dirs:
        if dir == "cross_validation":
            continue
        file_path = os.path.join(organ_dir, dir, str(file_name))
        dose_range, volume_predicted, volume = Get_Diff_DVHs(model, file_path)
        volumes_predicted.append(volume_predicted)
        volumes.append(volume)
    total_volumes = np.linspace(0, 0, 1000)
    total_volumes_pred = np.linspace(0, 0, 1000) 
    for volume_idx in range(len(volumes)):
        total_volumes += volumes[volume_idx] / len(subseg_dirs)
        total_volumes_pred += volumes_predicted[volume_idx] / len(subseg_dirs)
    total_volumes = NormalizeDVH(dose_range, total_volumes)
    total_volumes_pred = NormalizeDVH(dose_range, total_volumes_pred)   
    total_volumes_cumul = GetCumulativeDVH(dose_range, total_volumes)
    total_volumes_pred_cumul = GetCumulativeDVH(dose_range, total_volumes_pred)
    #first plot these:
    fig, axs = plt.subplots(2,2)
    colors = ['r', 'b', 'g', 'c', 'm', 'lime', 'darkorange']
    for dvh_idx in range(len(volumes_predicted)):
        c = colors[dvh_idx % 7]
        axs[0,0].plot(dose_range, volumes_predicted[dvh_idx], label=f"S{dvh_idx}", color=c)
        axs[0,1].plot(dose_range, volumes[dvh_idx], label=f"S{dvh_idx}", color=c)
    axs[0,0].set_title("Predicted Differential DVHs")
    axs[0,0].set_ylabel("Volume")
    axs[0,0].set_xlabel("Normalized Dose")   
    axs[0,0].legend() 
    axs[0,1].set_ylabel("Volume")
    axs[0,1].set_xlabel("Normalized Dose")
    axs[0,1].set_title("Planned Differential DVHs")
    axs[0,1].legend()

    axs[1,0].plot(dose_range, total_volumes_pred, label="Predicted Differential DVHs")
    axs[1,0].set_ylabel("Volume")
    axs[1,0].set_xlabel("Normalized Dose")
    axs[1,0].plot(dose_range, total_volumes, label="Planned Differential DVHs")
    axs[1,0].set_title("Differential DVHs")
    axs[1,0].legend()

    axs[1,1].plot(dose_range, total_volumes_pred_cumul, label="Predicted Cumulative DVHs")
    axs[1,1].set_ylabel("Volume")
    axs[1,1].set_xlabel("Normalized Dose")
    axs[1,1].plot(dose_range, total_volumes_cumul, label="Planned Cumulative DVHs")
    axs[1,1].set_title("Cumulative DVHs")
    axs[1,1].legend()
    plt.show()
    print("")   

def GetCumulativeDVH(dose_range, volumes):
    volume = 1
    cumulative_range = np.linspace(0,0, 1000)
    for idx in range(len(dose_range)-1):
        volume -= volumes[idx] * (dose_range[idx+1]-dose_range[idx])
        cumulative_range[idx] = volume
    return cumulative_range    


def NormalizeDVH(dose_range, volume_vals):
    sum = 0
    for idx in range(len(dose_range)-1):
        sum += (dose_range[idx+1]-dose_range[idx]) * volume_vals[idx]
    return volume_vals / sum    

def Get_Diff_DVHs(model, file_path):
    #first get the predicted parameters


    stack = np.zeros((1, 176))
    stack_y = np.zeros((1, 2))


    idx = 0
    data = pickle.load(open(file_path, "rb"))
    stack[idx,:] = data[0]
    stack_y[idx,:] = data[1]
    idx += 1


    set = DataSet(stack,stack_y)
    loader = torch.utils.data.DataLoader(set, batch_size=1, shuffle=False, num_workers=1)  
   
    for i, data in enumerate(loader):    #get % diffs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()

        outputs = model(inputs)
        outputs = outputs.detach().numpy()
        predicted_params = ([outputs[0][0], outputs[0][1]])
        params =([targets[0][0], targets[0][1]])
    
    #now get the curves for the different dvhs using the integral equation
    dose_range = np.linspace(0, 1.2, 1000)
    integral_equation = np.vectorize(DVH_Fitting.skew_normal_integral_all)
    volume_vals_predicted = integral_equation(dose_range, predicted_params[0], predicted_params[1], 0)
    volume_vals = integral_equation(dose_range, params[0], params[1], 0)
    volume_vals = NormalizeDVH(dose_range, volume_vals)
    volume_vals_predicted = NormalizeDVH(dose_range, volume_vals_predicted)
    return dose_range, volume_vals_predicted, volume_vals





