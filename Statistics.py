import os
import numpy as np
from shapely.geometry import Polygon, Point, LineString, polygon
from math import cos, sin, pi
import pickle 
from Contours import Contours
from Patient import Patient
import statistics as stats
import Contour_Operations

patients_path = os.path.join(os.getcwd(), "Patients")
processed_path = os.path.join(os.getcwd(), "Processed_Patients")

def Scale_NonExisting_Features(X_training_stack, y_training_stack , X_validation_stack, y_validation_stack):
    num_training_files = X_training_stack.shape[0]
    num_validation_files = X_validation_stack.shape[0]   
    num_features_X = X_training_stack.shape[1] 
    num_outputs_y = y_training_stack.shape[1]
    num_channels_X = X_training_stack.shape[2]
    num_channels_y = y_training_stack.shape[2]
    new_none_values = np.zeros((num_features_X, num_channels_X))
    #Now for each feature and each channel, I need to go through and collect all values that aren't 1000,
    # and then scale features that are 1000 to the maximum + 1 std
    for f in range(num_features_X):
        for c in range(num_channels_X):
            values = []
            for s, sample in enumerate(X_training_stack):
                if X_training_stack[s, f, c] != 1000:
                    values.append(X_training_stack[s, f, c])
            for s, sample in enumerate(X_validation_stack):
                if X_validation_stack[s, f, c] != 1000:
                    values.append(X_validation_stack[s, f, c])    
            std = stats.pstdev(values)
            max_val = max(values)
            #now go back through and make every value that is 1000 be max + std
            new_none_val = round(max_val + std, 1)      
            new_none_values[f,c] = new_none_val
            for s, sample in enumerate(X_training_stack):
                if X_training_stack[s, f, c] == 1000:
                    X_training_stack[s, f, c] = new_none_val
            for s, sample in enumerate(X_validation_stack):
                if X_validation_stack[s, f, c] == 1000:
                    X_validation_stack[s, f, c] = new_none_val  


    #Now do this for the y vals.
    for f in range(num_outputs_y):
        for c in range(num_channels_y):
            values = []
            for s, sample in enumerate(y_training_stack):
                if y_training_stack[s, f, c] != 1000:
                    values.append(y_training_stack[s, f, c])
            for s, sample in enumerate(y_validation_stack):
                if y_validation_stack[s, f, c] != 1000:
                    values.append(y_validation_stack[s, f, c])    
            std = stats.pstdev(values)
            max_val = max(values)
            #just make 1000 --> 0 for now for predictions. (best?)
            new_none_val = round(max_val + std, 1)       
            for s, sample in enumerate(y_training_stack):
                if y_training_stack[s, f, c] == 1000:
                    y_training_stack[s, f, c] = 0 #new_none_val
            for s, sample in enumerate(y_validation_stack):
                if y_validation_stack[s, f, c] == 1000:
                    y_validation_stack[s, f, c] = 0
    print("Rescaled non-existing features.") 
    #save the new none values
    with open (os.path.join(os.getcwd(), "Saved_Values", "new_none_values.txt"), "wb") as fp:
        pickle.dump(new_none_values, fp)
    return X_training_stack, y_training_stack , X_validation_stack, y_validation_stack              



def Get_ROI_Frequencies(file, occurrences_dict):
    try:
        with open(os.path.join(processed_path, file), "rb") as fp:
            patient : Patient = pickle.load(fp)
    except:
        return occurrences_dict  

    if getattr(patient, "ptv70") == None:
        return occurrences_dict      
    #now such rois exist. 
    for key in occurrences_dict:
        obj = getattr(patient, key)  
        if obj != None:
            occurrences_dict[key] += 1   
    return occurrences_dict
#this displays how many occurrences there are for each ROI in the patient cohort, for deciding what to use for model building





 


if __name__ == "__main__":
    poly = Contour_to_Polygon([[0,0,0] , [0,1,0], [1,1,3], [1,0,2]])
    poly2 = Contour_to_Polygon([[3,2,0] , [3,4,0], [4,4,3], [4,3,2]])
    centre = [-0.5, 0.5]
    point = Point(centre)
    distance = Point_to_Polygon_Distance(point, poly2)
    distance2 = Polygon_to_Polygon_Distance(poly, poly2)
    poly = Polygon()
    contour1 = [[[4,4,0] , [4,5,0], [5,5,3], [5,4,2]], [[1,0,0] , [0,1,0], [1,1,3], [1,0,2]]]
    contour2 = [[[3,2,0] , [3,4,0], [4,3,3], [4,3.5,2]],[[5,2,1] , [5,4,0], [9,4,3], [7,0,2]]]
    contourDist = Distance_Between_ROIs(contour1, contour2)
    print("")

        



