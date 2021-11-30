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


patients_path = os.path.join(os.getcwd(), "Patients")
processed_path = os.path.join(os.getcwd(), "Processed_Patients")
training_Path = os.path.join(os.getcwd(), "Training_Data")

def Get_HN_Patients():

    organs = [
        "brain",
        "brainstem",
        "brachial_plexus",
        "chiasm",
        "cochlea", 
        "pharyngeal_constrictors",
        "esophagus",
        "lacrimal_glands", 
        "larynx",
        "lens", 
        "lips", 
        "mandible", 
        "optic_nerves",
        "oral_cavity",
        "parotid_left",
        "parotid_right", 
        "spinal_cord", 
        "submandibular_right",
        "submandibular_left", 
        "thyroid", 
        "retina"
    ]
    ptvs = ["30", "35", "40", "45", "50", "54", "55", "60", "70", "63", "56"] 

    training_list = ["brainstem", "larynx", "mandible", "oral_cavity", "parotid_left", 
                    "parotid_right", "spinal_cord", "submandibular_right", "submandibular_left"]

    roi_list = CloneList(organs)
    for i in range(len(ptvs)):
        ptvs[i] = str("ptv" + str(ptvs[i]))
    roi_list.extend(CloneList(ptvs))
    patients = os.listdir(patients_path)
    processed_Files = os.listdir(processed_path)
    occurrences_dict = dict.fromkeys(roi_list, 0)
    # for patient in patients:
    #     print("Getting OARs for " + patient)
    #     patient_path = os.path.join(patients_path, patient)
        # dose_array = Get_Dose_Array(patient, patient_path)
        # if dose_array is None:
        #    continue
        # GetContours(patient, patient_path, organs)
        # GetPTVs(patient, patient_path, ptvs)
        # Get_ROI_to_PTV_Distances(patient, organs, ptvs)
        #Get_DVHs(patient, organs, ptvs, dose_array)
        #Now need the dose stats
        #print("")
    for file in processed_Files:    
        occurrences_dict = Statistics.Get_ROI_Frequencies(file, occurrences_dict)
        Get_Training_Data(file, training_list)
    print("Finished processing data for head and neck patients")

def Get_Training_Data(file, roi_list):
    X = np.ones((12,3)) * 1000 #9 oars, 3 ptv distances for each
    #First, return without saving if there is no ptv70. 
    # Last 3 features is volume dose for ptv56,63,70
    y = np.ones((9, 5)) * 1000 #9 rois, 6 dose features each
    try:
        with open(os.path.join(processed_path, file), "rb") as fp:
            patient : Patient = pickle.load(fp)
    except:
        return 
    if getattr(patient, "ptv70") is None:
        return

    for i, oar in enumerate(roi_list):
        org : Contours = getattr(patient, oar)
        if org != None:
            ptv70_dist = float(1000 if org.ptv70_dist is None else round(org.ptv70_dist,2))
            ptv63_dist = float(1000 if org.ptv63_dist is None else round(org.ptv63_dist,2))
            ptv56_dist = float(1000 if org.ptv56_dist is None else round(org.ptv56_dist,2))
            dose_vals = [1000]*5 if org.dose is None else org.dose
            dose_vals = list(dose_vals.values())[1:]
            dose_vals = [round(float(j),2) for j in dose_vals]
            X[i, :] = np.array([ptv70_dist, ptv63_dist, ptv56_dist]) 
            y[i,:] = np.array(dose_vals)  
    
    for p, ptv in enumerate(["56", "63", "70"]):
        ptv : Contours= getattr(patient, str("ptv" + ptv))
        if ptv is None:
            continue
        ptv_volume_dose = 1000 if ptv.volume_dose is None else ptv.volume_dose
        ptv_volume_dose = list(ptv_volume_dose.values())
        ptv_volume_dose = [round(float(val),2) for val in ptv_volume_dose]
        X[p+9,:] = np.array(ptv_volume_dose)

    save_path = os.path.join(training_Path, file)
    with open(save_path, "wb") as fp:
        pickle.dump([X, y], fp)
    print("Finished getting training data for " + file)

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
        print("Getting masks for " + organ)
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
            print("")
            setattr(organ_obj, "dose", dose)
            setattr(patient, organ, organ_obj)        


    for ptv in ptvs:
        ptv_obj = getattr(patient, str("ptv" + str(ptv)))  
        if ptv_obj == None:
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
            print("")
            setattr(ptv_obj, "dose", dose)
            

            #for ptvs I also want V95, V97, V99.
            volume_dose = Get_PTV_Volume_Doses(ptv_dose_pixels, float(ptv))
            setattr(ptv_obj, "volume_dose", volume_dose)
            setattr(patient, str("ptv" + str(ptv)), ptv_obj)

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
    print("")           

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
    if len(files) <= 2:
        files = glob.glob(os.path.join(files[0],  "*.dcm"))
    else:
        temp = CloneList(files)
        files = []
        for file in temp:
            if "dump" not in file.lower():
                files.append(file)


    for file in files:
        patientData = pydicom.dcmread(file)
        modality = patientData[0x0008,0x0060].value 
        if "DOSE" in modality:
            doseFile = patientData
            break
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
                full_array[1,z,y_idx,x_idx] = x
                full_array[2,z,y_idx,x_idx] = y   
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
            ptv_obj = getattr(patient, str("ptv" + str(ptv)))  
            if ptv_obj == None:
                continue
            else:
                ptv_contours = ptv_obj.wholeROI
            dist = Statistics.Distance_Between_ROIs(organ_contours, ptv_contours)
            print("distance between " + str(organ) + " and " + str(ptv) + ": " + str(dist))
            setattr(organ_obj, str("ptv" + ptv + "_dist"), dist)
            setattr(patient, organ, organ_obj)
        with open(os.path.join(processed_patient_path, patient_name), "wb") as fp:
                pickle.dump(patient, fp)
            


            


    print("")

def GetPTVs(patient, patient_path, ptvs): 

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
    for ptv in ptvs:
        structures, structure_roi_nums, struct_idxs = FindPTVs(structFiles, ptv)    
        if structure_roi_nums != 1111:
            noPTVs = False
        else: continue    
            
        contourList = []
        for idx in range(len(structures)):
            structure = structures[idx]
            structure_roi_num = structure_roi_nums[idx]
            struct_idx = struct_idxs[idx]
            
            structsMeta = pydicom.dcmread(structFiles[struct_idx]).data_element("ROIContourSequence")                    
            for contourInfo in structsMeta:
                if contourInfo.get("ReferencedROINumber") == structure_roi_num: #get the matched contour for the given organ
                    print(str("saving PTV" + ptv + " contours to " + patient + ". Matched structure: " + structure))
                    
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

                        contours = Contours(str("ptv" + str(ptv)), structure, contourList)   
                        #now save contours to patient object. 
                        processed_patient_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Research/Processed_Patients"
                        if os.path.exists(os.path.join(processed_patient_path, patient)):
                            try:
                                with open(os.path.join(processed_patient_path, patient), "rb") as fp:
                                    processed_patient = pickle.load(fp)
                            except:
                                processed_patient = Patient(patient, str(os.path.join(patient_path, patient)))    
                        else:
                            processed_patient = Patient(patient, str(os.path.join(patient_path, patient)))    

                        try:
                            setattr(processed_patient, str("ptv" + str(ptv)) , contours)   
                        except:
                            print("No contours saved to file.")
                        #save
                        with open(os.path.join(processed_patient_path, patient), "wb") as fp:
                            pickle.dump(processed_patient, fp)      
                    except: 

                        print("No contour Sequence.")
    if noPTVs:
        print("PTV" + str(ptv) + " not found.")
        Print_all_PTVs(structFiles)

 
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
    found = False
    for fileNum, file in enumerate(structureList):        
        roiSequence = pydicom.dcmread(file).data_element("StructureSetROISequence")
        for element in roiSequence:
            if "ptv" in element.get("ROIName").lower() and str(ptv) in element.get("ROIName").lower():
                print("Found PTV" + str(ptv) + " as " + element.get("ROIName"))
                found = True
                roiNumber = element.get("ROINumber")
                names.append(element.get("ROIName").lower())
                roiNums.append(roiNumber)
                fileNums.append(fileNum)
    
    if found == False:
        return [], 1111, 0
    return names, roiNums, fileNums    

def GetContours(patient, patient_path, organs): 
    #save contour list for specified organ for all patients to patient binary file

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
        structure, structure_roi_num, struct_idx = FindStructure(structFiles, organ)
        #confirm match
        inp = ""
        # if structure != "":
        #     while True:
        #         try:
        #             inp = input(patient + ": matched structure for " + organ + " is " + structure + ". Is this correct? y/n")
        #             if inp == "y":
        #                 print("accepted.")
        #                 break    
        #             elif inp == "n":
        #                 print("rejected")
        #                 break             
        #         except KeyboardInterrupt:
        #             quit()
        #         except: pass 
        #If a structure isn't found, deal with it accordingly    
        # if structure_roi_num == 1111 or inp == "n":
        #     roi_options = []
        #     roi_options_names = []
        #     struct_files_idx = []
        #     for struct_idx, struct in enumerate(structFiles):
        #         roiSequence = pydicom.dcmread(struct).data_element("StructureSetROISequence")               
        #         for element in roiSequence:
        #             i = element.get("ROINumber")
        #             roi_options.append(i)
        #             roi_options_names.append(element.get("ROIName").lower())
        #             struct_files_idx.append(struct_idx)
        #             print(str(i+1) + ": " + element.get("ROIName").lower())
        #     print("Could not automatically find " + organ + ". Continue or choose an organ option or press 0 to continue without.")

            # while True:
            #     try:
            #         inp = input("Enter an option: \n")
            #         if int(inp) == 0:
            #             print(organ + " does not exist. skipping.")
            #             break
            #         if int(inp)-1 in roi_options:
            #             struct_idx = struct_files_idx[roi_options.index(int(inp)-1)] 
            #             structure_roi_num = int(inp)
            #             structure = roi_options_names[roi_options.index(int(inp)-1)] 
            #             break        

            #     except KeyboardInterrupt:
            #         quit()
            #     except: pass 
        if structure_roi_num == 1111:    
            print("No structure found.")
            continue #nothing to save to the patient
            # else:
            #     print("Continuing with organ: " + str(structure))

        structsMeta = pydicom.dcmread(structFiles[struct_idx]).data_element("ROIContourSequence")        
        contourList = []
        for contourInfo in structsMeta:
            if contourInfo.get("ReferencedROINumber") == structure_roi_num: #get the matched contour for the given organ
                print(str("saving " + organ + " contours to " + patient + ". Matched structure: " + structure))
                
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
                    
                    contours = Contours(organ, structure, contourList)   
                    #now save contours to patient object. 
                    processed_patient_path = "//PHSAhome1.phsabc.ehcnet.ca/csample1/Profile/Desktop/Research/Processed_Patients"
                    if os.path.exists(os.path.join(processed_patient_path, patient)):
                        try:
                            with open(os.path.join(processed_patient_path, patient), "rb") as fp:
                                processed_patient = pickle.load(fp)
                        except:
                            processed_patient = Patient(patient, str(os.path.join(patient_path, patient)))    
                    else:
                        processed_patient = Patient(patient, str(os.path.join(patient_path, patient)))    

                    try:
                        setattr(processed_patient, organ, contours)   
                    except:
                        print("No contours saved to file.")
                    #save
                    with open(os.path.join(processed_patient_path, patient), "wb") as fp:
                        pickle.dump(processed_patient, fp)
                except: 
                    print("No contour Sequence.")
                
                


                
def FindStructure(structureList, organ, invalidStructures = []):
    """Finds the matching structure to a given organ in a patient's
       dicom file metadata. 
     
    Args:
        structureList (List): a list of paths to RTSTRUCT files in the patient folder
        organ (str): the organ to find the matching structure for
        invaidStructures (list): a list of structures that the matching 
            structure cannot be, defaults to an empty list

    Returns: 
        str, int: the matching structure's name in the metadata, the 
            matching structure's ROI number in the metadata. Returns "", 
            1111 if no matching structure is found in the metadata
        
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
                    return element.get("ROIName").lower(), roiNumber, fileNum
                if element.get("ROIName").lower() == "sm_r" and organ  == "Right Submandibular":
                    roiNumber = element.get("ROINumber")
                    return element.get("ROIName").lower(), roiNumber, fileNum    
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
        return "", 1111, 0    
    #Now return the organ that is remaining and has closest string
    fileNum = -1
    for file in structureList:
        fileNum = fileNum + 1
        roiSequence = pydicom.dcmread(file).data_element("StructureSetROISequence")
        for element in roiSequence:
            if element.get("ROIName").lower() == closestStrings[0][0]:
                roiNumber = element.get("ROINumber")
    try:
        return closestStrings[0][0], roiNumber, fileNum
    except:
        return "", 1111, 0 #error code for unfound match.                                                                                                          

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
    if ("_l_" in s1) or (" l " in s1) or  (" l-" in s1) or ("-l-" in s1) or (" l_" in s1) or ("_l " in s1) or ("-l " in s1) or ("left" in s1) or ("l " == s1[0:2]) or ("_lt_" in s1) or (" lt " in s1) or  (" lt-" in s1) or ("-lt-" in s1) or (" lt_" in s1) or ("_lt " in s1) or ("-lt " in s1) or ("lt " == s1[0:3]) or ("_l" == s1[-2:]):
        if not (("lpar" in s2) or ("lsub" in s2) or ("_l_" in s2) or (" l " in s2) or  (" l-" in s2) or ("-l-" in s2) or (" l_" in s2) or ("_l " in s2) or ("-l " in s2) or ("left" in s2) or ("l " == s2[0:2])or ("_lt_" in s2) or (" lt " in s2) or  (" lt-" in s2) or ("-lt-" in s2) or (" lt_" in s2) or ("_lt " in s2) or ("-lt " in s2) or ("lt " == s2[0:3]) or ("_l" == s2[-2:])):   
            allowed = False  
    if (("_l_" in s2) or (" l " in s2) or  (" l-" in s2) or ("-l-" in s2) or (" l_" in s2) or ("_l " in s2) or ("-l " in s2) or ("left" in s2) or ("l " == s2[0:2])or ("_lt_" in s2) or (" lt " in s2) or  (" lt-" in s2) or ("-lt-" in s2) or (" lt_" in s2) or ("_lt " in s2) or ("-lt " in s2)or ("lt " == s2[0:3]) or ("_l" == s2[-2:])):  
        if not (("lpar" in s1) or ("lsub" in s1) or ("_l_" in s1) or (" l " in s1) or  (" l-" in s1) or ("-l-" in s1) or (" l_" in s1) or ("_l " in s1) or ("-l " in s1) or ("left" in s1) or ("l " == s1[0:2]) or ("_lt_" in s1) or (" lt " in s1) or  (" lt-" in s1) or ("-lt-" in s1) or (" lt_" in s1) or ("_lt " in s1) or ("-lt " in s1) or ("lt " == s1[0:3]) or ("_l" == s1[-2:])):
            allowed = False        
    
    if ("_r_" in s1) or (" r " in s1) or  (" r-" in s1) or ("-r-" in s1) or (" r_" in s1) or ("_r " in s1) or ("-r " in s1) or ("right" in s1) or ("r " == s1[0:2])or ("_rt_" in s1) or (" rt " in s1) or  (" rt-" in s1) or ("-rt-" in s1) or (" rt_" in s1) or ("_rt " in s1) or ("-rt " in s1)or ("right" in s1) or ("_r" == s1[-2:]):
        if not (("rpar" in s2) or ("rsub" in s2) or ("_r_" in s2) or (" r " in s2) or  (" r-" in s2) or ("-r-" in s2) or (" r_" in s2) or ("_r " in s2) or ("-r " in s2) or ("right" in s2) or ("r " == s2[0:2]) or ("_rt_" in s2) or (" rt " in s2) or  (" rt-" in s2) or ("-rt-" in s2) or (" rt_" in s2) or ("_rt " in s2) or ("-rt" in s2) or ("_r" == s2[-2:])):   
            allowed = False
    if (("_r_" in s2) or (" r " in s2) or  (" r-" in s2) or ("-r-" in s2) or (" r_" in s2) or ("_r " in s2) or ("-r " in s2) or ("right" in s2) or ("r " == s2[0:2]) or ("_rt_" in s2) or (" rt " in s2) or  (" rt-" in s2) or ("-rt-" in s2) or (" rt_" in s2) or ("_rt " in s2) or ("-rt" in s2) or ("_r" == s2[-2:])): 
        if not (("rpar" in s1) or ("rsub" in s1) or ("_r_" in s1) or (" r " in s1) or  (" r-" in s1) or ("-r-" in s1) or (" r_" in s1) or ("_r " in s1) or ("-r " in s1) or ("right" in s1) or ("r " == s1[0:2])or ("_rt_" in s1) or (" rt " in s1) or  (" rt-" in s1) or ("-rt-" in s1) or (" rt_" in s1) or ("_rt " in s1) or ("-rt " in s1) or ("_r" == s1[-2:])):
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

