from concurrent.futures import process
from numpy.core.numeric import full
import pydicom 
import numpy as np
import os
import glob
from fastDamerauLevenshtein import damerauLevenshtein
import pickle
from operator import itemgetter
from Contours import Contours
from Patient import Patient
import Statistics 
import copy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from shapely.geometry.polygon import Polygon
import random
import re
import Chopper
import Visuals
import statistics
import Distance_Operations
import DVH_Fitting
import scipy as sp

try:
    a = 2/0
    patients_path = os.path.join(os.getcwd(), "Patients")
    processed_path = os.path.join(os.getcwd(), "Processed_Patients")
    training_Path = os.path.join(os.getcwd(), "Training_Data")
    statistics_path = os.path.join(os.getcwd(), "Statistics")
except: 
    patients_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Programs/Dose_Score_Function_old/20211110_Caleb_SGFX"    
    processed_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Programs/Dose_Score_Function/Processed_Patients"
    training_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Programs/Dose_Score_Function/Processed_Patients/Training_Data"
    statistics_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Programs/Dose_Score_Function/Statistics"

#--------------------------------------------------------------------------------------------------------------------------------------------


def Get_HN_Patients():

    organs = [
        "brainstem",
        "larynx",
        "oral_cavity",
        "parotid_left",
        "parotid_right", 
        "spinal_cord", 
    ]


    # patients = os.listdir(patients_path)
    # #random.shuffle(patients)
    
    # for patient in patients:
    #     print("Getting OARs for " + patient)

    #     patient_path = os.path.join(patients_path, patient)
    #     processed_patient_path = os.path.join(os.getcwd(), "Processed_Patients")
    #     processed_patient = GetContours(patient, patient_path, organs, try_load = False)
        
    #     if processed_patient == 1: #invalid
    #         continue
    #     if processed_patient.prescription_dose == None or processed_patient.dose_array == []:
    #         continue
    #     processed_patient = Distance_Operations.Get_Spatial_Relationships(processed_patient)
    #     processed_patient = Distance_Operations.Get_OAR_Distances(organs, processed_patient)
    #     #save data
    #     processed_patient = Distance_Operations.Get_OAR_Distances(organs, processed_patient)
    #     with open(os.path.join(processed_patient_path, patient), "wb") as fp:
    #         pickle.dump(processed_patient, fp)
            

    #     #Now need the dose stats
    #     print(f"Finished collecting data for {patient}")

    Distance_Operations.Get_Distance_Stats(processed_path, statistics_path, organs)
    print("Finished processing data for head and neck patients")

def Get_Training_Data(file, roi_list):
    #Calculating training data. 
    print("Finished getting training data for " + file)

def Get_Dose_Voxels(dose_array, contours_obj: Contours):
    #first get whole roi
    contours = contours_obj.wholeROI

    #get max, min x and y
    min_x = 1000
    max_x = -1000
    min_y = 1000
    max_y = -1000
    min_z = 1000
    max_z = -1000
    for slices in contours:
        for slice in slices:
            if slice == []:
                continue

            if slice[0][2] < min_z:
                min_z = slice[0][2]
            if slice[0][2] > max_z:
                max_z = slice[0][2]   

            for point in slice:
                if point[0] < min_x:
                    min_x = point[0]
                if point[0] > max_x:
                    max_x = point[0]
                if point[1] < min_y:
                    min_y = point[1]
                if point[1] > max_y:
                    max_y = point[1]  
   
    #now crop the dose array
    x_vals = dose_array[1, 0, 0,:]
    y_vals = dose_array[2, 0, :,0]
    z_vals = dose_array[3,:,0,0]

    x_bounds = [0,len(x_vals)]
    y_bounds = [0,len(y_vals)]
    z_bounds = [0, len(z_vals)]

    if z_vals[0] > z_vals[-1]:
        dose_array = dose_array[:, ::-1,:,:]

    for x in range(dose_array.shape[3]):
        if dose_array[1,0,0,x] > min_x:
            x_bounds[0] = max(x-1,0)
            for x_2 in range(x, dose_array.shape[3]):    
                if dose_array[1,0,0,x_2] > max_x:
                    x_bounds[1] = x_2
                    break
            break        

    for y in range(dose_array.shape[2]):
        if dose_array[2,0,y,0] > min_y:
            y_bounds[0] = max(y-1,0)    
            for y2 in range(y, dose_array.shape[2]):
                if dose_array[2,0,y2,0] > max_y:
                    y_bounds[1] = y2   
                    break
            break
    for z in range(dose_array.shape[1]):
        if dose_array[3,z,0,0] > min_z:
            z_bounds[0] = max(0,z-1)    
            for z2 in range(z, dose_array.shape[1]):
                if dose_array[3,z2,0,0] > max_z:
                    z_bounds[1] = z2  
                    break
            break    
    #now crop dose array
    cropped_array =  dose_array[:,z_bounds[0]:z_bounds[1], y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]           
    #now upsample array
    upsampled_array = cropped_array.repeat(3, axis=3).repeat(3, axis=2)



        
    # contours = contours_obj.wholeROI #first get for whole ROI
    organ_masks = GetContourMasks(contours, upsampled_array)
    organ_dose_voxels = []
    for slice in range(upsampled_array.shape[1]):
        if np.amax(organ_masks[0,slice,:,:]) == 0: #no contour on slice
            continue
        layer_bool_mask = organ_masks[0, slice, :, :] > 0
        contour_dose = upsampled_array[0, slice, :,:][layer_bool_mask]
        #contour_dose is a list of all dose pixels in contour on current slice.
        organ_dose_voxels.extend(contour_dose)
        #now divide organ dose pixels into 10 bins. 
    organ_dose_voxels.sort()
    contours_obj.dose_voxels = copy.deepcopy(organ_dose_voxels)

    subseg_dose_voxels = []    
    for subseg in contours_obj.segmentedContours:
        organ_masks = GetContourMasks(subseg, upsampled_array)
        organ_dose_voxels = []
        for slice in range(upsampled_array.shape[1]):
            if np.amax(organ_masks[0,slice,:,:]) == 0: #no contour on slice
                continue
            layer_bool_mask = organ_masks[0, slice, :, :] > 0
            contour_dose = upsampled_array[0, slice, :,:][layer_bool_mask]
            #contour_dose is a list of all dose pixels in contour on current slice.
            organ_dose_voxels.extend(contour_dose)
            #now divide organ dose pixels into 10 bins. 
        organ_dose_voxels.sort()
        subseg_dose_voxels.append(copy.deepcopy(organ_dose_voxels))
    
    contours_obj.dose_voxels_subsegs = subseg_dose_voxels


def Get_DVHs(patient_name, organs, ptvs, dose_array):    
    processed_patient_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Research/Processed_Patients"
    if os.path.exists(os.path.join(processed_patient_path, patient_name)):
        try:
            with open(os.path.join(processed_patient_path, patient_name), "rb") as fp:
                patient : Patient = pickle.load(fp)
        except:
            raise Exception("Processed patient object for " + patient_name + " was not loadable. Quitting.") 
    else:     
        print("Processed patient object for " + patient_name + " was not found. Skipping")    
        return

    #Now one by one get the distances between existing ROIs and PTVs.
    for organ in organs:
        organ_obj = getattr(patient, organ)  
        if organ_obj == None:
            continue
        organ_contours = organ_obj.wholeROI
        organ_masks = GetContourMasks(organ_contours, dose_array)
        organ_dose_pixels = []
        for slice in range(dose_array.shape[1]):
            if np.amax(organ_masks[0,slice,:,:]) == 0: #no contour on slice
                continue
            layer_bool_mask = organ_masks[0, slice, :, :] > 0
            contour_dose = dose_array[0, slice, :,:][layer_bool_mask]
            #contour_dose is a list of all dose pixels in contour on current slice.
            organ_dose_pixels.extend(contour_dose)
        #now divide organ dose pixels into 10 bins. 
        organ_dose_pixels.sort()
        #Now i want to save Dmin, D50, D75, D90, D95, Dmax
        if len(organ_dose_pixels) > 20:

            d_min = min(organ_dose_pixels)
            d_max = max(organ_dose_pixels)
            dose_bins = Get_DVH_Bins(organ_dose_pixels)
            #Each of these bins is 5% of the volume.
            d_50 = dose_bins[10][0]
            d_75 =  dose_bins[15][0]
            d_90 = dose_bins[18][0]
            d_95 = dose_bins[19][0]
            dose = {
                "d_min": d_min,
                "d_max": d_max,
                "d_50": d_50,
                "d_75": d_75,
                "d_90": d_90,
                "d_95": d_95
            }
            setattr(organ_obj, "dose", dose)
            setattr(patient, organ, organ_obj)        

    patient_ptvs = patient.PTVs
    for ptv in ptvs:
        try:
            ptv_obj = patient_ptvs[ str("ptv" + str(ptv))]
        except:
            continue
        ptv_contours = ptv_obj.wholeROI
        print("Getting masks for " + ptv)
        ptv_masks = GetContourMasks(ptv_contours, dose_array)
        ptv_dose_pixels = []
        for slice in range(dose_array.shape[1]):
            if np.amax(ptv_masks[0,slice,:,:]) == 0: #no contour on slice
                continue
            layer_bool_mask = ptv_masks[0, slice, :, :] > 0
            contour_dose = dose_array[0, slice, :,:][layer_bool_mask]
            #contour_dose is a list of all dose pixels in contour on current slice.
            ptv_dose_pixels.extend(contour_dose)
        #now divide organ dose pixels into 10 bins. 
        ptv_dose_pixels.sort()
        #Now i want to save Dmin, D50, D75, D90, D95, Dmax
        if len(organ_dose_pixels) > 20:
            d_min = min(ptv_dose_pixels)
            d_max = max(ptv_dose_pixels)
            dose_bins = Get_DVH_Bins(ptv_dose_pixels)
            #Each of these bins is 5% of the volume.
            d_50 = dose_bins[10][0]
            d_75 =  dose_bins[15][0]
            d_90 = dose_bins[18][0]
            d_95 = dose_bins[19][0]
            dose = {
                "d_min": d_min,
                "d_max": d_max,
                "d_50": d_50,
                "d_75": d_75,
                "d_90": d_90,
                "d_95": d_95
            }

            setattr(ptv_obj, "dose", dose)
            

            #for ptvs I also want V95, V97, V99.
            volume_dose = Get_PTV_Volume_Doses(ptv_dose_pixels, float(ptv))
            setattr(ptv_obj, "volume_dose", volume_dose)
            patient.PTVs[str("ptv" + str(ptv))] = ptv_obj

    try:        
        with open(os.path.join(processed_patient_path, patient_name), "wb") as fp:
            pickle.dump(patient, fp)
    except: 
        raise Exception("Could not save processed patient with new DVH attribute.") 
         
    if os.path.exists(os.path.join(processed_patient_path, patient_name)):
        try:
            with open(os.path.join(processed_patient_path, patient_name), "rb") as fp:
                patient : Patient = pickle.load(fp)
        except:
            raise Exception("Processed patient object for " + patient_name + " was not loadable. Quitting.") 
    else:     
        print("Processed patient object for " + patient_name + " was not found. Skipping")    
        return 
        

def Get_PTV_Volume_Doses(ptv_dose_pixels, ptv_dose):
    perc_95 = 0.95 * ptv_dose
    perc_97 = 0.97 * ptv_dose
    perc_99 = 0.99 * ptv_dose
    
    total_pixels = len(ptv_dose_pixels)
    pixels_above_95 = len(list(filter (lambda d: d > perc_95, ptv_dose_pixels)))
    pixels_above_97 = len(list(filter (lambda d: d > perc_97, ptv_dose_pixels)))
    pixels_above_99 = len(list(filter (lambda d: d > perc_99, ptv_dose_pixels)))

    v95 = round((pixels_above_95 / total_pixels), 3) * 100
    v97 = round((pixels_above_97 / total_pixels), 3) * 100
    v99 = round((pixels_above_99 / total_pixels), 3) * 100
    volume_dose = {
        "v_95": v95,
        "v_97": v97,
        "v_99": v99
    }
    return volume_dose


def Get_DVH_Bins(list, num_bins=20):
    list_len = len(list)
    bin_size = int(list_len / num_bins)
    remainder = list_len % 20
    new_list = []
    list_iterator = iter(list)
    for i in range(num_bins):
        new_list.append([])
        for j in range(bin_size):
            new_list[i].append(next(list_iterator))
        if remainder:
            new_list[i].append(next(list_iterator))
            remainder -= 1
    return new_list





              

def Get_Dose_Array(patient, patient_path):
    print("Getting dose array for " + patient)
    files = glob.glob(os.path.join(patient_path, "*"))
    
    
    #sometimes there is a nested study directory here instead of the file list. in that case need to move in one more level
    doseFile = None
    if len(files) <= 2:
        files = glob.glob(os.path.join(files[0],  "*.dcm"))
    else:
        temp = CloneList(files)
        files = []
        for file in temp:
            if "dump" not in file.lower():
                files.append(file)


    for file in files:
        try:
            patientData = pydicom.dcmread(file)
            modality = patientData[0x0008,0x0060].value 
            if "DOSE" in modality:
                doseFile = patientData
                break
        except:
            continue 
    if doseFile == None:
        print("No dose array found for " + patient)
        return []       
    dose_array = patientData.pixel_array * float(patientData[0x3004, 0x000E].value) #Dose Grid Scaling Attribute
    dose_units = patientData[0x3004, 0x0002].value
    if dose_units != "GY":
        raise Exception("Dose array was not in units of gy.")
    pixel_spacing = patientData[0x0028,0x0030].value
    ipp = patientData[0x0020,0x0032].value #image position patient
    iop = patientData[0x0020, 0x0037].value
    if iop != [1,0,0,0,1,0]:
        raise Exception("IOP is non standard.")
    grid_offset_vector = patientData[0x3004, 0x000C].value
    if grid_offset_vector[0] != 0:
        raise Exception("Grid offset vector is not relative.")
    zValues = []
    for vec in grid_offset_vector:
        zValues.append(vec + ipp[2])
    #print(dose_array.shape)   
    full_array = np.zeros((4, dose_array.shape[0], dose_array.shape[1], dose_array.shape[2]))
    for z in range(len(zValues)):
        full_array[3, z, :, :] = zValues[z]
        full_array[0,z,:,:] = dose_array[z,:,:]   
        # plt.imshow(full_array[0,z,:,:]) 
        # plt.show()                
    for x_idx in range(dose_array.shape[2]):
        for y_idx in range(dose_array.shape[1]):
            x = ipp[0] + x_idx*pixel_spacing[0]
            y = ipp[1] + y_idx*pixel_spacing[1]
            full_array[1,:,y_idx,x_idx] = x
            full_array[2,:,y_idx,x_idx] = y   
    #print(np.amax(full_array[0,:,:,:]))        
    return full_array   
        
def GetContourMasks(contours, Array):
    numImagesSlices, len1, len2 = Array.shape[1:]
    contourMasks = np.zeros((2,numImagesSlices,len1,len2)) #2 channels, one for filled and one for unfilled
    unshown=True
    contours = CartesianToPixelCoordinates(CloneList(contours), Array)
    for idx in range(numImagesSlices):#loop through all slices creating a mask for the contours
        for contour in contours:
            for island in contour:
                if len(island) < 3:
                    continue
                if abs(int(round(island[0][2], 2)*100) - int(round(Array[3,idx,0,0], 2)*100)) < 2: #if contour is on the current slice
                    contourMaskFilled = Image.new('L', (len2, len1), 0 )
                    contourMaskUnfilled = Image.new('L', (len2, len1), 0 )
                    contourPoints = []
                    for point in island:
                        contourPoints.append((int(point[0]), int(point[1]))) #changed
                    contourPolygon = Polygon(contourPoints)
                    ImageDraw.Draw(contourMaskFilled).polygon(contourPoints, outline= 1, fill = 1)
                    ImageDraw.Draw(contourMaskUnfilled).polygon(contourPoints, outline= 1, fill = 0)
                    contourMaskFilled = np.array(contourMaskFilled)
                    contourMaskUnfilled = np.array(contourMaskUnfilled)
                    # if np.amax(contourMaskUnfilled) != 0 and unshown==True:
                    #     plt.imshow(contourMaskFilled)
                    #     plt.show()
                    #     unshown=False
                    #     print("")
                    contourMasks[0, idx, :,:] = contourMaskFilled
                    contourMasks[1,idx,:,:] = contourMaskUnfilled    
                    break                     
    return contourMasks                

def CloneList(list):
    listCopy = copy.deepcopy(list)
    return listCopy  

def CartesianToPixelCoordinates(contours, array):
    #convert x and y values for a contour into the pixel indices where they are on the pet array
    xVals = array[1,0,0,:]
    yVals = array[2,0,:,0]
    for contour in contours: 
        for island in contour:
            for point in island:
                point[0] = min(range(len(xVals)), key=lambda i: abs(xVals[i]-point[0]))
                point[1] = min(range(len(yVals)), key=lambda i: abs(yVals[i]-point[1]))
    return contours  


def Get_ROI_to_PTV_Distances(patient_name, organs, ptvs):
    processed_patient_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Research/Processed_Patients"
    if os.path.exists(os.path.join(processed_patient_path, patient_name)):
            try:
                with open(os.path.join(processed_patient_path, patient_name), "rb") as fp:
                    patient : Patient = pickle.load(fp)
            except:
                raise Exception("Processed patient object for " + patient_name + " was not loadable. Quitting.") 
    else:     
        print("Processed patient object for " + patient_name + " was not found. Skipping")    
        return

    #Now one by one get the distances between existing ROIs and PTVs.
    for organ in organs:
        organ_obj = getattr(patient, organ)  
        if organ_obj == None:
            continue
        else:
            organ_contours = organ_obj.wholeROI
        for ptv in ptvs:
            try:
                ptv_obj = patient.PTVs[str("ptv" + str(ptv))]
            except:
                continue
            else:
                ptv_contours = ptv_obj.wholeROI
            dist = Statistics.Distance_Between_ROIs(organ_contours, ptv_contours)
            print("distance between " + str(organ) + " and " + str(ptv) + ": " + str(dist))
            setattr(organ_obj, str("ptv" + ptv + "_dist"), dist)
            setattr(patient, organ, organ_obj)
        with open(os.path.join(processed_patient_path, patient_name), "wb") as fp:
                pickle.dump(patient, fp)
            


            


def GetPTVs(processed_patient, patient, patient_path): 
    processed_patient_path = os.path.join(os.getcwd(),"Processed_Patients")

        
    files = glob.glob(os.path.join(patient_path, "*"))
    structFiles = [] 
    #sometimes there is a nested study directory here instead of the file list. in that case need to move in one more level
    if len(files) <= 2:
        files = glob.glob(os.path.join(files[0],  "*.dcm"))
    else:
        files = glob.glob(os.path.join(patient_path, "*.dcm"))

    for file in files:
        patientData = pydicom.dcmread(file)
        modality = patientData[0x0008,0x0060].value 
        if "STRUCT" in modality:
            structFiles.append(file)   
    noPTVs = True    
    ptv_prescriptions = []    
    for ptv in range(20, 71):
        structures, structure_roi_nums, struct_idxs, ptv_types = FindPTVs(structFiles, ptv)    
        if structure_roi_nums == 1111:
            continue
        else:
            noPTVs=False
        ptv_prescriptions.append(ptv)    
        contourList = []
        for idx in range(len(structures)):
            structure = structures[idx]
            ptv_type = ptv_types[idx]
            structure_roi_num = structure_roi_nums[idx]
            struct_idx = struct_idxs[idx]
            
            structsMeta = pydicom.dcmread(structFiles[struct_idx]).data_element("ROIContourSequence")                    
            for contourInfo in structsMeta:
                if contourInfo.get("ReferencedROINumber") == structure_roi_num: #get the matched contour for the given organ
                    print(str("saving PTV" + str(ptv) + " contours to " + patient + ". Matched structure: " + structure + ", type = " + ptv_type))
                    
                    try: #sometimes in this dataset, contoursequence dne
                        for contoursequence in contourInfo.ContourSequence: 
                            contour_data = contoursequence.ContourData
                            #But this is easier to work with if we convert from a 1d to a 2d list for contours ( [ [x1,y1,z1], [x2,y2,z2] ... ] )
                            tempContour = []
                            i = 0
                            if len(contour_data) > 3:
                                z = float(contour_data[2])
                            while i < len(contour_data):
                                x = float(contour_data[i])
                                y = float(contour_data[i + 1])                             
                                tempContour.append([x, y, z ])
                                i += 3     
                            #first look to see if a contour at z value already exists
                            slice_exists = False
                            for element in contourList:
                                try:
                                    if element[0][0][2] == z: #check z value of first point in first island of slice
                                        element.append(tempContour)     
                                        slice_exists = True
                                        break                                 
                                                                   
                                except:
                                    print("Warning: error occurred when checking if ROI already had a contour at the z value: " + str(z))
                            if slice_exists == False:        
                                contourList.append([tempContour])  

                        ptv_name = str("ptv" + str(ptv))
                        # if ptv_type != "reg":    #if want to keep l and r ptvs separate
                        #     ptv_name += str("_" + ptv_type)
                        contours = Contours(ptv_name, structure, contourList)   
                        #now save contours to patient object. 
                         

                        try:
                            processed_patient.PTVs.setdefault(ptv_name, []).append(contours)  
                        except:
                            print("No contours saved to file.")
                        
                    except: 
                        print("No contour Sequence.")
    if noPTVs:
        print("PTV" + str(ptv) + " not found.")
        Print_all_PTVs(structFiles)
        processed_patient.prescription_dose = None
    else:    
        processed_patient.prescription_dose = max(ptv_prescriptions)    
       
    return processed_patient

 
def Print_all_PTVs(structureList):
    print("All PTVs found in structure file: ")
    for fileNum, file in enumerate(structureList):        
        roiSequence = pydicom.dcmread(file).data_element("StructureSetROISequence")
        for element in roiSequence:
            if "ptv" in element.get("ROIName").lower():
                print(element.get("ROIName").lower())

def FindPTVs(structureList, ptv):
    #can return multiple ptvs, so keep a list for each
    names = []
    roiNums = []
    fileNums = []
    ptv_types = []
    found = False
    for fileNum, file in enumerate(structureList):        
        roiSequence = pydicom.dcmread(file).data_element("StructureSetROISequence")
        for element in roiSequence:
            ptv_name = element.get("ROIName").lower()
            #print(element.get("ROIName").lower())
            allowed, ptv_type = AllowedToMatchPTVS(ptv_name, str(ptv))
            if allowed==True:
                #don't take ptv if "all" in it, because its a duplicate
                print(f"Found PTV {ptv} as {ptv_name}. Type = {ptv_type}")
                found = True
                roiNumber = element.get("ROINumber")
                names.append(ptv_name)
                roiNums.append(roiNumber)
                fileNums.append(fileNum)
                ptv_types.append(ptv_type)

                if "all" in ptv_name:
                    #if combined boolean ptv then dont need to save other ptvs
                    print(f"Found combined \"all\" PTV{ptv}.")
                    names.clear()
                    roiNums.clear()
                    fileNums.clear()

                    names.append(ptv_name)
                    roiNums.append(roiNumber)
                    fileNums.append(fileNum)
                    ptv_types.append(ptv_type)

                    return names, roiNums, fileNums, ptv_types  
    
    if found == False:
        return [], 1111, 0, "reg"
    return names, roiNums, fileNums, ptv_types   


def Print_DICOM_Structures(structureList):
    for fileNum, file in enumerate(structureList):        
        roiSequence = pydicom.dcmread(file).data_element("StructureSetROISequence")
        for element in roiSequence:
            name = element.get("ROIName").lower()
            print(name.lower())
def Filter_Contour_Slice_Ends(contours):
    #look for last contour that is far away from other slices   
    filtered_contours = [] 
    for slice in contours:
        if len(slice[0]) != 0:
            filtered_contours.append(copy.deepcopy(slice))
    if abs(filtered_contours[-1][0][0][2] - filtered_contours[-2][0][0][2]) > 5:
        del filtered_contours[-1]
    if abs(filtered_contours[0][0][0][2] - filtered_contours[1][0][0][2]) > 5:
        del filtered_contours[-1]    
    return filtered_contours    


def GetContours(patient, patient_path, organs, try_load=False): 
    #save contour list for specified organ for all patients to patient binary file
    processed_patient_path = os.path.join(os.getcwd(), "Processed_Patients")

    if try_load == True:
        if os.path.exists(os.path.join(processed_patient_path, patient)):
            try:
                with open(os.path.join(processed_patient_path, patient), "rb") as fp:
                    processed_patient = pickle.load(fp)
                    return processed_patient
            except:
                processed_patient = Patient(patient, str(os.path.join(patient_path, patient)))    
        else:
            processed_patient = Patient(patient, str(os.path.join(patient_path, patient))) 

    else: 
        processed_patient = Patient(patient, str(os.path.join(patient_path, patient))) 

    processed_patient = GetPTVs(processed_patient, patient, patient_path)    
    if len(processed_patient.PTVs) > 3 or len(processed_patient.PTVs) == 0:
        print("Skipping patient because 0 or more than 3 ptv types found")
        return 1
    #get the patient's dose array 
    processed_patient.dose_array = Get_Dose_Array(patient, patient_path)  
    if processed_patient.dose_array == []:
        return processed_patient
    files = glob.glob(os.path.join(patient_path, "*"))
    structFiles = [] 
    #sometimes there is a nested study directory here instead of the file list. in that case need to move in one more level
    if len(files) <= 2:
        files = glob.glob(os.path.join(files[0],  "*.dcm"))
    else:
        files = glob.glob(os.path.join(patient_path, "*.dcm"))

    for file in files:
        patientData = pydicom.dcmread(file)
        modality = patientData[0x0008,0x0060].value 
        if "STRUCT" in modality:
            structFiles.append(file)  

    for organ in organs:         
        print("Looking for " + organ + "...")    
        structure, structure_roi_num, struct_idx, dicom_names = FindStructure(structFiles, organ)

        

        if "stem" in organ:    #only save attribute once, when looking for brain-stem.
            processed_patient.dicom_structures = dicom_names
            print("Dicom structures:")
            for val in dicom_names:
                print(val)
        if structure_roi_num == 1111:    
            print(f"{organ} not found. Ending processing.")
            return 1    #error, skip patient
        else:
            print(f"{organ} matched with {structure}")        
            

        structsMeta = pydicom.dcmread(structFiles[struct_idx]).data_element("ROIContourSequence")        
        contourList = []
        for contourInfo in structsMeta:
            if contourInfo.get("ReferencedROINumber") == structure_roi_num: #get the matched contour for the given organ
                
                
                try: #sometimes in this dataset, contoursequence dne
                    for contoursequence in contourInfo.ContourSequence: 
                        contour_data = contoursequence.ContourData
                except: 

                    print("No contour Sequence.")
                    return 1

                for contoursequence in contourInfo.ContourSequence: 
                    contour_data = contoursequence.ContourData
                    tempContour = []
                    i = 0
                    if len(contour_data) > 3: 
                        z = float(contour_data[2])
                    while i < len(contour_data):
                        x = float(contour_data[i])
                        y = float(contour_data[i + 1])                             
                        tempContour.append([x, y, z ])
                        i += 3    
                    #first look to see if a contour at z value already exists
                    slice_exists = False
                    for element in contourList:
                        try:
                            if element[0][0][2] == z: #check z value of first point in first island of slice
                                element.append(tempContour)     
                                #add list as another island at slice if slice exists
                                slice_exists = True
                                
                                break                                 
                                                            
                        except:
                            print("Warning: error occurred when checking if ROI already had a contour at the z value: " + str(z))
                    if slice_exists == False:        
                        contourList.append([tempContour]) 
                contourList = Filter_Contour_Slice_Ends(contourList)
                contours = Contours(organ, structure, contourList)    #creat contours object 
                print("Segmenting contours...")

                #perform segmentation
                #
                #if organ is spinal cord, then segmentation is different (divided into 2cm chunks...)
                if organ=="spinal_cord":
                    Chopper.CordChopper(contours)
                elif "submand" in organ:    
                    Chopper.OrganChopper(contours, [1,1,1])
                elif "mandible" in organ:
                    Chopper.MandibleChopper(contours)         
                else:    
                    Chopper.OrganChopper(contours, [2,2,1])
                Get_Dose_Voxels(processed_patient.dose_array, contours)  
                if processed_patient.prescription_dose == None:
                    continue
                #now save contours to patient object. 
                dvh_arrays, dvh_params = DVH_Fitting.Get_DVH(contours.dose_voxels, processed_patient.prescription_dose)  
                setattr(contours, "dvh_bins_whole" , dvh_arrays)   
                setattr(contours, "dvh_params_whole" , dvh_params)   

                dvh_arrays_subsegs = []
                dvh_params_subsegs = []
                for dose_array in contours.dose_voxels_subsegs:
                    if len(dose_array) == 0:
                        dvh_arrays_subsegs.append([])
                        dvh_params_subsegs.append([])
                        continue

                    dvh_arrays, dvh_params = DVH_Fitting.Get_DVH(dose_array, processed_patient.prescription_dose)  
                    dvh_arrays_subsegs.append(dvh_arrays)
                    dvh_params_subsegs.append(dvh_params)

                setattr(contours, "dvh_bins_subsegs" , dvh_arrays_subsegs)   
                setattr(contours, "dvh_params_subsegs" , dvh_params_subsegs)      
                    
                try:
                    setattr(processed_patient, organ, contours)   
                except:
                    print("No contours saved to file.")
                #save
                    

    # #also look for the opti structures
    # for organ in organs:         
    #     print("Looking for opti structure for " + organ + " ...")    
    #     structure, structure_roi_num, struct_idx, dicom_names = FindStructure(structFiles, str(organ + "_opti"))
    #     #confirm match
    #     inp = ""
    #     if structure_roi_num == 1111:    
    #         print(f"{organ} not found.")
    #         continue #nothing to save to the patient
    #         # else:
    #         #     print("Continuing with organ: " + str(structure))

    #     structsMeta = pydicom.dcmread(structFiles[struct_idx]).data_element("ROIContourSequence")        
    #     contourList = []
    #     for contourInfo in structsMeta:
    #         if contourInfo.get("ReferencedROINumber") == structure_roi_num: #get the matched contour for the given organ
    #             print(str("saving " + organ + " opti contours to " + patient + ". Matched structure: " + structure))
                
    #             try: #sometimes in this dataset, contoursequence dne
    #                 for contoursequence in contourInfo.ContourSequence: 
    #                     contour_data = contoursequence.ContourData
    #                     #But this is easier to work with if we convert from a 1d to a 2d list for contours ( [ [x1,y1,z1], [x2,y2,z2] ... ] )

    #                     tempContour = []
    #                     i = 0
    #                     if len(contour_data) > 3:
    #                         z = float(contour_data[2])
    #                     while i < len(contour_data):
    #                         x = float(contour_data[i])
    #                         y = float(contour_data[i + 1])                             
    #                         tempContour.append([x, y, z ])
    #                         i += 3    
    #                     #first look to see if a contour at z value already exists
    #                     slice_exists = False
    #                     for element in contourList:
    #                         try:
    #                             if element[0][0][2] == z: #check z value of first point in first island of slice
    #                                 element.append(tempContour)     
    #                                 slice_exists = True
    #                                 break                                 
                                                                
    #                         except:
    #                             print("Warning: error occurred when checking if ROI already had a contour at the z value: " + str(z))
    #                     if slice_exists == False:        
    #                         contourList.append([tempContour]) 
                    
    #                 contours = Contours(organ, structure, contourList)   
    #                 #now save contours to patient object.                        
    #                 try:
    #                     setattr(processed_patient, str("opti_" + organ), contours)   
    #                 except:
    #                     print("No contours saved to file.")

    #             except: 
    #                 print("No contour Sequence.")                
    #                 print(str("saving " + organ + " contours to " + patient + ". Matched structure: " + structure))
    # # with open(os.path.join(processed_patient_path, patient), "wb") as fp:
    # #     pickle.dump(processed_patient, fp) 
    return processed_patient                
                
                


                
def FindStructure(structureList, organ, invalidStructures = []):
    """Finds the matching structure to a given organ in a patient's
       dicom file metadata. 
     
    Args:
        structureList (List): a list of paths to RTSTRUCT files in the patient folder
        organ (str): the organ to find the matching structure for
        invaidStructures (list): a list of structures that the matching 
            structure cannot be, defaults to an empty list

    Returns: 
        str, int, int, list: the matching structure's name in the metadata, the 
            matching structure's ROI number in the metadata, the index indicating which structure file has the structure, and a list of all dicom structure names found. Returns "", 
            1111, 0, list if no matching structure is found in the metadata
        
    """

    #Here we take the string for the desired structure (organ) and find the matching structure for each patient. 
    #The algorithm is to first make sure that the organ has a substring matching the ROI with at least 3 characters,
    #then out of the remaining possiblities, find top 3 closest fits with damerau levenshtein algorithm, then check to make sure that they are allowed to match according to rules defined in AllowedToMatch(). There should then ideally
    # be only one possible match, but if there are two, take the first in the list.   

    #Get a list of all structures in structure set: 
    unfilteredStructures = []

    for fileNum, file in enumerate(structureList):
        
        roiSequence = pydicom.dcmread(file).data_element("StructureSetROISequence")
        for element in roiSequence:
            if element.get("ROIName").lower() not in invalidStructures:
                #first check if need to do special matching for limbus name:
                if element.get("ROIName").lower() == "sm_l" and organ  == "Left Submandibular":
                    roiNumber = element.get("ROINumber")
                    return element.get("ROIName").lower(), roiNumber, fileNum, []
                if element.get("ROIName").lower() == "sm_r" and organ  == "Right Submandibular":
                    roiNumber = element.get("ROINumber")
                    return element.get("ROIName").lower(), roiNumber, fileNum, []    
                
                unfilteredStructures.append(element.get("ROIName").lower())  
               
    #Now find which is the best fit.
    #First filter out any structures without at least a 3 character substring in common
    structures = []
    for structure in unfilteredStructures:
        valid = True
        if LongestSubstring(structure, organ) < 3:
            valid = False
        if not AllowedToMatch(organ, structure): 
            valid = False
        #Add to structures if valid
        if valid:
            structures.append(structure)
    #Now test string closeness to find
    closestStrings = [["",100],["",100],["",100]] #has to be in the top 3 closest strings to check next conditions
    for structure in structures:
        closeness = StringDistance(structure, organ)
        closestStrings.sort(key=itemgetter(1)) #Sort by the closeness value, and not the structure names
        for value in range(len(closestStrings)):
            if closeness < closestStrings[value][1]: #If closer than a value already in the top 3
                closestStrings[value] = [structure, closeness]
                break
    
    if len(closestStrings) == 0:
        return "", 1111, 0, unfilteredStructures   
    #Now return the organ that is remaining and has closest string
    fileNum = -1
    for file in structureList:
        fileNum = fileNum + 1
        roiSequence = pydicom.dcmread(file).data_element("StructureSetROISequence")
        for element in roiSequence:
            if element.get("ROIName").lower() == closestStrings[0][0]:
                roiNumber = element.get("ROINumber")
    try:
        return closestStrings[0][0], roiNumber, fileNum, unfilteredStructures
    except:
        return "", 1111, 0, unfilteredStructures #error code for unfound match.                                                                                                          

def AllowedToMatchPTVS(s1, ptv):
    allowed=True
    ptv_type = "reg"
    s1 = s1.lower()
    ptv = ptv.lower()
    if "ptv" not in s1 or ptv not in s1:
        allowed=False
        return False, ptv_type
    
    keywords = []
    keywords.append("opti")
    keywords.append("all")
    keywords.append("comb")
    for keyword in keywords:
        if keyword in s1:
            return False, ptv_type 
 
    #its tricky matching up left and right organs sometimes with all the conventions used... this makes sure that both are left or both are right
    if (("_l_" in s1) or (" l " in s1) or  (" l-" in s1) or ("-l-" in s1) or (" l_" in s1) or ("_l " in s1) or ("-l " in s1) or ("left" in s1) or ("l " == s1[0:2]) or ("_lt_" in s1) or 
        (" lt " in s1) or  (" lt-" in s1) or ("-lt-" in s1) or (" lt_" in s1) or ("_lt " in s1) or ("-lt " in s1) or ("lt " == s1[0:3]) or ("_l" == s1[-2:]) or (re.search("\d[-_]?l ", s1) != None) or (s1[-1] == "l" and s1[-2].isdigit())):
        ptv_type = "L"
      
    if ("_r_" in s1) or (" r " in s1) or  (" r-" in s1) or ("-r-" in s1) or (" r_" in s1) or ("_r " in s1) or ("-r " in s1) or ("right" in s1) or ("r " == s1[0:2])or ("_rt_" in s1) or (" rt " in s1) or  (" rt-" in s1) or ("-rt-" in s1) or (" rt_" in s1) or ("_rt " in s1) or ("-rt " in s1)or ("right" in s1) or ("_r" == s1[-2:]) or (re.search("\d[-_]?r ", s1) != None) or (s1[-1] == "r" and s1[-2].isdigit()):
        ptv_type = "R"
    return allowed, ptv_type        

def AllowedToMatch(s1, s2):
    """Determines whether or not s1 and s2 are allowed to match 
       based on if they both contain the correct substrings.
     
    Args:
        s1 (str): first string to determine match
        s2 (str): second string to determine match

    Returns: 
        allowed (bool): True if the strings are allowed to match, 
            false otherwise
        
    """

    s1 = s1.lower()
    s2 = s2.lower()
    allowed = True
    keywords = []
    #You can't have only one organ with one of these keywords...
    keywords.append("prv")
    keywords.append("tub")
    keywords.append("brain")
    keywords.append("ptv")
    keywords.append("stem")
    keywords.append("node")
    keywords.append("cord")
    keywords.append("chi")
    keywords.append("opt")
    keywords.append("oral")
    keywords.append("nerv")
    keywords.append("par")
    keywords.append("globe")
    keywords.append("lip")
    keywords.append("cav")
    keywords.append("sub")
    keywords.append("test")
    keywords.append("fact")
    keywords.append("lacrim")
    keywords.append("constrict")
    keywords.append("esoph")
    keywords.append("couch")
    keywords.append("gtv")
    keywords.append("ctv")
    keywords.append("avoid")
    keywords.append("fact")

    
    #keywords can't be in only one of two string names: 
    for keyword in keywords:
        num = 0
        if keyword in s1:
            num += 1
        if keyword in s2:
            num += 1
        if num == 1:
            allowed = False        

    #Cant have left and no l in other, or right and no r
    if "left" in s1:
        if "l" not in s2:
            allowed = False      
    if "left" in s2:
        if "l" not in s1:
            allowed = False    
    #its tricky matching up left and right organs sometimes with all the conventions used... this makes sure that both are left or both are right
    if (("_l_" in s1) or (" l " in s1) or  (" l-" in s1) or ("-l-" in s1) or (" l_" in s1) or ("_l " in s1) or ("-l " in s1) or ("left" in s1) or ("l " == s1[0:2]) or ("_lt_" in s1) or 
        (" lt " in s1) or  (" lt-" in s1) or ("-lt-" in s1) or (" lt_" in s1) or ("_lt " in s1) or ("-lt " in s1) or ("lt " == s1[0:3]) or ("_l" == s1[-2:]) or (re.search("\d[-_]?l ", s1) != None)):
        if not (("lpar" in s2) or ("lsub" in s2) or ("_l_" in s2) or (" l " in s2) or  (" l-" in s2) or ("-l-" in s2) or (" l_" in s2) or ("_l " in s2) or ("-l " in s2) or ("left" in s2) or 
            ("l " == s2[0:2])or ("_lt_" in s2) or (" lt " in s2) or  (" lt-" in s2) or ("-lt-" in s2) or (" lt_" in s2) or ("_lt " in s2) or ("-lt " in s2) or ("lt " == s2[0:3]) or ("_l" == s2[-2:]) or (re.search("\d[-_]?l ", s2) != None)):   
            allowed = False  
    if (("_l_" in s2) or (" l " in s2) or  (" l-" in s2) or ("-l-" in s2) or (" l_" in s2) or ("_l " in s2) or ("-l " in s2) or ("left" in s2) or ("l " == s2[0:2])or ("_lt_" in s2) or (" lt " in s2) 
    or  (" lt-" in s2) or ("-lt-" in s2) or (" lt_" in s2) or ("_lt " in s2) or ("-lt " in s2)or ("lt " == s2[0:3]) or ("_l" == s2[-2:]) or (re.search("\d[-_]?l ", s2))):  
        if not (("lpar" in s1) or ("lsub" in s1) or ("_l_" in s1) or (" l " in s1) or  (" l-" in s1) or ("-l-" in s1) or (" l_" in s1) or ("_l " in s1) or ("-l " in s1) or ("left" in s1) or ("l " == s1[0:2]) or 
        ("_lt_" in s1) or (" lt " in s1) or  (" lt-" in s1) or ("-lt-" in s1) or (" lt_" in s1) or ("_lt " in s1) or ("-lt " in s1) or ("lt " == s1[0:3]) or ("_l" == s1[-2:]) or (re.search("\d[-_]?l ", s1))):
            allowed = False        
    
    if ("_r_" in s1) or (" r " in s1) or  (" r-" in s1) or ("-r-" in s1) or (" r_" in s1) or ("_r " in s1) or ("-r " in s1) or ("right" in s1) or ("r " == s1[0:2])or ("_rt_" in s1) or (" rt " in s1) or  (" rt-" in s1) or ("-rt-" in s1) or (" rt_" in s1) or ("_rt " in s1) or ("-rt " in s1)or ("right" in s1) or ("_r" == s1[-2:]) or (re.search("\d[-_]?r ", s1)):
        if not (("rpar" in s2) or ("rsub" in s2) or ("_r_" in s2) or (" r " in s2) or  (" r-" in s2) or ("-r-" in s2) or (" r_" in s2) or ("_r " in s2) or ("-r " in s2) or ("right" in s2) or ("r " == s2[0:2]) or ("_rt_" in s2) or (" rt " in s2) or  (" rt-" in s2) or ("-rt-" in s2) or (" rt_" in s2) or ("_rt " in s2) or ("-rt" in s2) or ("_r" == s2[-2:]) or (re.search("\d[-_]?r ", s2))):   
            allowed = False
    if (("_r_" in s2) or (" r " in s2) or  (" r-" in s2) or ("-r-" in s2) or (" r_" in s2) or ("_r " in s2) or ("-r " in s2) or ("right" in s2) or ("r " == s2[0:2]) or ("_rt_" in s2) or (" rt " in s2) or  (" rt-" in s2) or ("-rt-" in s2) or (" rt_" in s2) or ("_rt " in s2) or ("-rt" in s2) or ("_r" == s2[-2:]) or (re.search("\d[-_]?r ", s2))): 
        if not (("rpar" in s1) or ("rsub" in s1) or ("_r_" in s1) or (" r " in s1) or  (" r-" in s1) or ("-r-" in s1) or (" r_" in s1) or ("_r " in s1) or ("-r " in s1) or ("right" in s1) or ("r " == s1[0:2])or ("_rt_" in s1) or (" rt " in s1) or  (" rt-" in s1) or ("-rt-" in s1) or (" rt_" in s1) or ("_rt " in s1) or ("-rt " in s1) or ("_r" == s1[-2:]) or (re.search("\d[-_]?r ", s1))):
            allowed = False
    return allowed


def StringDistance(s1, s2):
    """returns the Damerau-Levenshtein distance between two strings

    Args:
        s1 (string): string one which is to be compared with string 2.
        s2 (string): string two which is to be compared with string 1.

    Returns:
        (int): the Damerau Levenshtein distance between s1 and s2, which indicates how different the two strings are in terms of the amount of deletion, insertion, substitution, and transposition operations required to equate the two.

    """
    return damerauLevenshtein(s1,s2,similarity=False)

def LongestSubstring(s1,s2):
    """Finds the length of the longest substring that is in 
       both s1 and s2.
     
    Args:
        s1 (str): the first string to find the longest substring in
        s2 (str): the second string to find the longest substring in

    Returns: 
        longest (int): the length of the longest substring that is in 
            s1 and s2
        
    """

    m = len(s1)
    n = len(s2)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(s1[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(s1[i-c+1:i+1])
    return longest   

