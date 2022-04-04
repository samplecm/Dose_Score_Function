import Contours 
from Contour_Operations import Contour_to_Polygon, Interpolate_Slices
import Statistics
import statistics 
import Patient
import Distance_Operations
import math
import Chopper
import copy
from shapely.geometry import Polygon, Point, LineString, polygon
import logging
logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

def Get_Spatial_Relationships(patient_obj : Patient):
    #need to get overlap fraction, and spatial proximities of all ptvs to each OAR.
    #format spatial data as numpy array:
    #[
    # PTV types (rel) - 3 boxes, 0 if any dne
    # PTV min distances (all 18 spatial regions)
    # PTV max distances (all 18 spatial regions)
    # overlap fractions
    # oar distances
    # ]
    
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

    ptvs = patient_obj.PTVs
    ptv_types = []
    for key in ptvs:    #get max ptv in order to format ptvs as relative prescriptions
        ptv_types.append(int(key[-2:]))

    max_ptv = max(ptv_types)
    num_ptv_types = len(ptv_types)
    #first get centre point for oars and their subsegs
    for organ in organs:
        
        organ_obj = getattr(patient_obj, organ)
        if organ_obj == None:
            continue
        Get_Contour_Centres(organ_obj)      

    for organ in organs:
        print(f"Getting spatial relationships for {organ}")
        organ_obj = getattr(patient_obj, organ)  
        if organ_obj == None:
            continue
        spatial_data_subsegs = []
        for s, segment in enumerate(organ_obj.segmentedContours):
            spatial_data = []
            for p,ptv in enumerate(ptvs):
                spatial_data.append([])
                spatial_data[-1].append(ptv_types[p] / max_ptv)    
                ptv_list = ptvs[ptv]
                centre_point = organ_obj.centre_point_subsegs[s]
                overlap_frac, min_dists, max_dists  = Get_PTV_Distance(segment, ptv_list, centre_point)
                spatial_data[-1].append(overlap_frac)
                spatial_data[-1].append(min_dists)
                spatial_data[-1].append(max_dists)
            spatial_data_subsegs.append(spatial_data)
        organ_obj.spatial_data_subsegs = spatial_data_subsegs


        #get spatial data for each oar with each ptv:
    return patient_obj

def Get_PTV_Distance(roi : list, ptvs : list, centre_point ):
    #returns overlap fraction and the minimum and average distance to the PTV, in different windows divided with spherical coordinates. 
    #for each subsegment, I am interested in: 
    #radial distance to each ptv within 18 separate windows made of permutations of theta and phi divisions
    # theta:  -90 - -30, -30 - 30, 30 - 90 
    # phi: 0 - 60, 60- 120, 120-180, 180-240, 240 - 300, 300 - 360 
    ptv_arrays = []
    for ptv in ptvs:
        ptv_arrays.append(ptv.wholeROI)
    overlap_frac, centre_overlap_bool = Overlap_Frac(roi, ptv_arrays, centre_point)
    non_overlap_ptvs = Get_PTV_Bools(roi, ptv_arrays)
    pi = math.pi

    #make list for holding different angles 
    angle_points = [
        [[], [], [], [], [], []],
        [[], [], [], [], [], []],
        [[], [], [], [], [], []]
    ]

    #now loop over ptv points and get distance data
    centre_x = centre_point[0]
    centre_y = centre_point[1]
    centre_z = centre_point[2]

    thetas = []
    phis = []

    #first get list of ptv point angles:
    for ptv in non_overlap_ptvs:
        for slices in ptv:
            for slice in slices:
                for p, point in enumerate(slice):
                    if p % 2 != 0:  #use every second point
                        continue
                    if p == 0:
                        z = point[2]
                        delta_z = z - centre_z

                    x = point[0]
                    y =point[1]

                    delta_x = x - centre_x
                    delta_y = y - centre_y
                    if delta_x == 0: 
                        delta_x += 0.01
                    phi = math.atan(abs(delta_y)/abs(delta_x))
                    #fix quadrant
                    if delta_y < 0 and delta_x < 0:
                        phi += math.pi
                    elif delta_y >0 and delta_x < 0:
                        phi += math.pi     
                    if delta_z == 0:
                        theta = 0    
                    else:
                        theta = math.atan(math.sqrt(delta_x**2+delta_y**2)/delta_z) + math.pi/2    #in range 0 --> pi
                    thetas.append(theta)
                    phis.append(phi)    

    #now get angle ranges:
    thetas.sort() 
    phis.sort()
    theta_gap = [0,0, -100000] #hold angle1 and angle2 and gap which define the largest range where no points exist. 
    phi_gap = [0,0, -100000]
    for t in range(len(thetas)-1):
        if thetas[t+1] - thetas[t] > theta_gap[2]:
            theta_gap = [thetas[t], thetas[t+1], thetas[t+1] - thetas[t]]
    for p in range(len(phis)-1):
        if phis[p+1] - phis[p] > phi_gap[2]:
            phi_gap = [phis[p], phis[p+1], phis[p+1] - phis[p]]

    theta_range = [theta_gap[1], theta_gap[0]+ math.pi]
    phi_range = [phi_gap[1], phi_gap[0]+ 2*math.pi]

    if thetas[0] + (math.pi - thetas[-1]) > theta_gap[2]:
        theta_range = [thetas[0], thetas[-1]]
    if phis[0] + (2*math.pi - phis[-1]) > phi_gap[2]:
        phi_range = [phis[0], phis[-1]]

    phi_range_dist = (phi_range[1] - phi_range[0]) / 6
    theta_range_dist = (theta_range[1] - theta_range[0]) / 3

    for ptv in non_overlap_ptvs:
        for slices in ptv:
            for slice in slices:
                for p, point in enumerate(slice):
                    if p % 2 != 0:  #use every second point
                        continue
                    if p == 0:
                        z = point[2]
                        delta_z = z - centre_z

                    x = point[0]
                    y =point[1]

                    delta_x = x - centre_x
                    delta_y = y - centre_y
                    r = math.sqrt(delta_x**2 + delta_y **2 + delta_z **2 )
                    if delta_x == 0:
                        delta_x += 0.01
                    phi = math.atan(abs(delta_y)/abs(delta_x))
                    #fix quadrant
                    if delta_y < 0 and delta_x < 0:
                        phi += math.pi
                    elif delta_y >0 and delta_x < 0:
                        phi += math.pi     
                    if delta_z == 0:
                        theta = 0    
                    else:
                        theta = math.atan(math.sqrt(delta_x**2+delta_y**2)/delta_z) + math.pi /2
                    if phi < phi_range[0]:
                        phi += 2*math.pi
                    if theta < theta_range[0]:
                        theta += math.pi    
                    phi_bin = math.floor((phi - phi_range[0]) / phi_range_dist)
                    theta_bin = math.floor((theta - theta_range[0]) / theta_range_dist)
                    #handle case of the maximum angle points
                    if theta_bin == 3:
                        theta_bin = 2
                    if phi_bin == 6:
                        phi_bin = 5    

                    angle_points[theta_bin][phi_bin].append(r)

    min_dists = []
    max_dists = []            
    for theta_idx in range(len(angle_points)):
        for phi_idx in range(len(angle_points[theta_idx])):
            phi = phi_range[0] + phi_idx + phi_range_dist
            theta = theta_range[0] + theta_idx * theta_range_dist
            if centre_overlap_bool == True:
                min_dists.append([0, 0, 0])
                if angle_points[theta_idx][phi_idx] == []:
                    max_dists.append([1111, 1111, 1111])
                else:    
                    max_dists.append([max(angle_points[theta_idx][phi_idx]), phi, theta])

            elif angle_points[theta_idx][phi_idx] == []:
                min_dists.append([1111, 1111, 1111])
                max_dists.append([1111, 1111, 1111])

            else: 
                min_dists.append([min(angle_points[theta_idx][phi_idx]), phi, theta])    
                max_dists.append([max(angle_points[theta_idx][phi_idx]), phi, theta])
    # if overlap_frac > 0:
    #     print("hehe")
    return overlap_frac, min_dists, max_dists            

#--------------------------------------------------------------------------------------------------------------------------------------------

def Get_PTV_Bools(oar, ptv_list):
    #get ptv contour points that are boolean difference of oar points
    non_overlap_ptvs = []
    for ptv in ptv_list:
        non_overlap_ptvs.append([])
        
        for ptv_slices in ptv:
            oar_slice_exists = False
            non_overlap_ptvs[-1].append([])
            for ptv_slice in ptv_slices:
                ptv_z = ptv_slice[0][2]
                if len(ptv_slice) < 3:
                    continue
                for slice in oar:
                    slice = slice[0]
                    if len(slice) < 3:
                        continue
                    if slice[0][2] == ptv_z:
                        oar_slice_exists = True
                        ptv_pol = Contour_to_Polygon(ptv_slice)
                        oar_pol = Contour_to_Polygon(slice)  
                        try:
                            ptv_bool = ptv_pol.difference(oar_pol)
                        except:
                            ptv_bool = ptv_pol.buffer(0).difference(oar_pol.buffer(0))
                        try:
                            ptv_bool = ptv_bool.exterior.coords
                            non_overlap_ptvs[-1][-1].append([[point[0], point[1], ptv_z] for point in ptv_bool])  
                        except AttributeError:
                            ptv_bools = list(ptv_bool) 
                            for ptv_bool in ptv_bools:
                                ptv_bool_list = ptv_bool.exterior.coords
                                non_overlap_ptvs[-1][-1].append([[point[0], point[1], ptv_z] for point in ptv_bool_list])  

                        break
                        
                if oar_slice_exists == False:
                    non_overlap_ptvs[-1][-1].append(copy.deepcopy(ptv_slice))

    return non_overlap_ptvs


#--------------------------------------------------------------------------------------------------------------------------------------------

def Overlap_Frac(oar, ptv_list, centre_point):
    #return volume overlap fraction and a boolean for whether the centre point is in the ptv
    #also return list of non overlapping points
    oar_volume = 0
    opti_volume = 0

    
    centre_z = centre_point[2]
    centre_overlap_bool = False

    ptv_list_uninterpolated = copy.deepcopy(ptv_list)
    ptv_list = []
    #first interpolate any slices that are not in ptv (but not out of range)
    for ptv in ptv_list_uninterpolated:
        ptv_list.append(Interpolate_Slices(oar, ptv, centre_point[2]))

    for o in range(len(oar)-1):
        slice_1 = oar[o][0]
        slice_2 = oar[o+1][0]

        if len(slice_1) < 3 or len(slice_2) < 3:
            continue

        z_1 = slice_1[0][2]
        z_2 = slice_2[0][2]
        delta_z = abs(z_1 - z_2)

        oar_slice_pol_1 = Contour_to_Polygon(slice_1)            
        oar_slice_pol_2 = Contour_to_Polygon(slice_2) 

        pol_1_opti = oar_slice_pol_1
        pol_2_opti = oar_slice_pol_2     

        oar_volume += (oar_slice_pol_1.area + oar_slice_pol_2.area ) * 0.5 * delta_z 
 
        for ptv in ptv_list:
            for ptv_slices in ptv:
                for ptv_slice in ptv_slices:

                    if len(ptv_slice) < 3:
                        continue
                    if abs(int(round(centre_z, 2)*100) - int(round(ptv_slice[0][2], 2)*100)) < 2:
                        pol = Contour_to_Polygon(ptv_slice) 
                        point = Point(centre_point[0], centre_point[1])
                        if pol.contains(point):
                            centre_overlap_bool = True

                    if abs(int(round(z_1, 2)*100) - int(round(ptv_slice[0][2], 2)*100)) < 2:
                        ptv_slice_pol_1 = Contour_to_Polygon(ptv_slice) 
                        try:
                            pol_1_opti = pol_1_opti.difference(ptv_slice_pol_1)
                            #ptv_slice_pols_1.append(ptv_slice_pol_1.intersection(oar_slice_pol_1))   
                        except: 
                            pol_1_opti = pol_1_opti.buffer(0).difference(ptv_slice_pol_1.buffer(0))
                            #ptv_slice_pols_1.append(ptv_slice_pol_1.buffer(0).intersection(oar_slice_pol_1.buffer(0)))   #prevents self intersection error
                            pass          

                    if abs(int(round(z_2, 2)*100) - int(round(ptv_slice[0][2], 2)*100)) < 2:
                        ptv_slice_pol_2 = Contour_to_Polygon(ptv_slice)     
                        try:
                            pol_2_opti = pol_2_opti.difference(ptv_slice_pol_2)
                            #ptv_slice_pols_2.append(ptv_slice_pol_2.intersection(oar_slice_pol_2))   
                        except: 
                            pol_2_opti = pol_2_opti.buffer(0).difference(ptv_slice_pol_2.buffer(0))
                            #ptv_slice_pols_2.append(ptv_slice_pol_2.buffer(0).intersection(oar_slice_pol_2.buffer(0)))   
                            pass          
                        break  

        opti_volume += (pol_1_opti.area + pol_2_opti.area ) * 0.5 * delta_z 

   
    overlap_frac = (oar_volume - opti_volume) / oar_volume            
    overlap_frac = round(overlap_frac, 3)
    # if overlap_frac > 0:
    #     print("hehe")
    return overlap_frac, centre_overlap_bool      

        #get ptv at slice
#-------------------------------------------------------------------------------------------------------------------------------------------------

def Get_Contour_Centres(contours : Contours):
    print(f"Calculating centres for {contours.roiName}")

    #first calculate for whole ROI 
    centre_slice = Get_Centre_Slice(contours.wholeROI)    
    centre_point = Get_Centre_Point(centre_slice)

    #now calculate for all subsegments 
    subseg_centres = [] 
    for subsegment in contours.segmentedContours:
        centre_slice = Get_Centre_Slice(subsegment)
        centre_point = Get_Centre_Point(centre_slice)
        subseg_centres.append(centre_point)
    
    #set centres as contour attributes
    contours.centre_point = centre_point
    contours.centre_point_subsegs = subseg_centres


#----------------------------------------------------------------------------------------------------------------------------------------------

def Get_Centre_Point(centre_slice):
    #currently uses avg x,y for centre point
    x_vals = []
    y_vals = []
    for point in centre_slice[0]:
        x_vals.append(point[0])
        y_vals.append(point[1])
    x_avg = statistics.mean(x_vals)
    y_avg = statistics.mean(y_vals) 
    centre_point = [x_avg, y_avg, centre_slice[0][0][2]]
    return centre_point
        
#-----------------------------------------------------------------------------------------------------------------------------------------------

def Get_Centre_Slice(contours : list):
    z_vals = []
    for slice in contours:
        try:
            z_vals.append(slice[0][0][2])
        except IndexError: #contour empty on slice
            pass
     
    min_z = min(z_vals)
    max_z = max(z_vals)            
    centre_z = (max_z + min_z) * 0.5

    if centre_z in z_vals:
        centre_contour = copy.deepcopy(contours[z_vals.index(centre_z)])
    else: 
        #add slice to contours list
        contoursZ = Chopper.ClosestContourZ(centre_z, contours) #Get the closest contour above and below the cut
        newContour = []
        for c, contour in enumerate(contours[contoursZ[0]]):
            newContour.append([])
            try:
                closest_slice = contours[contoursZ[1]][c]
            except IndexError:
                continue 
            for j in range(len(contour)): #Go over all points in closest contour
                point1 = contour[j]      
                point2 = Chopper.ClosestPoint(point1, closest_slice, island_idx=True)    #Now get the closest point in second closest contour
                if point2 == None:
                    break
                newPoint = Chopper.InterpolateXY(point1, point2, centre_z)    #now interpolate between the two
                #add new point to new contour
                newContour[-1].append(newPoint)
            #add this new contour to newContoursList
        contours.append(newContour)    
         
        # z_diff = 10000
        # closest_z = 10000
        # closest_index = 1000
        # for s, slice in enumerate(contours): 
        #     if len(slice[0]) > 0:
        #         z = slice[0][0][2]
        #         if abs(centre_z - z) < z_diff:
        #             z_diff = abs(centre_z - z)
        #             closest_index = s            


        centre_contour = copy.deepcopy(newContour)   
        contours = list(filter(lambda layer: layer != [[]], contours))
        contours.sort(key=Chopper.GetZVal)  
    
    return centre_contour

