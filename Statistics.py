import os
import numpy as np
from shapely.geometry import Polygon, Point, LineString, polygon
from math import cos, sin, pi
import pickle 
from Contours import Contours
from Patient import Patient

patients_path = os.path.join(os.getcwd(), "Patients")
processed_path = os.path.join(os.getcwd(), "Processed_Patients")


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

def Point_to_Polygon_Distance(point : Point, polygon : Polygon):
    #this function retrieves the minimum distance between a point and a polygon
    #first check if the point is inside the polygon
    if point.within(polygon):
        return 0
    distance = polygon.distance(point)     
    # x = point.x
    # y = point.y
    # centre_x = polygon_centre[0]
    # centre_y = polygon_centre[1]
    # hyp = ((x-centre_x)**2 + (y-centre_y)**2)   
    # #first get the angle
    # cos_theta = (centre_x-x )/ hyp
    # sin_theta = (centre_y-y) / hyp
    # theta = np.arcsin(sin_theta)
    # if cos_theta < 0 and sin_theta > 0:
    #     theta = pi - theta
    # if cos_theta < 0 and sin_theta < 0:
    #     theta = pi + abs(theta) 
    # if cos_theta > 0 and sin_theta < 0:
    #     theta = 2 * pi - abs(theta)
    # line =  LineString([(x,y), (x + 10000 * cos_theta, y + 10000* sin_theta)])  
    #dist_min = None 
    # intersection = polygon.intersection(line)
    # if intersection.is_empty:
    #     raise Exception("Error: Line from point to centre of mass did not intersect the contour.")
    # elif intersection.geom_type.startswith('Multi') or intersection.geom_type  == 'GeometryCollection':
    #     point = min(intersection)
    # else:
    #     point = intersection    
    # distance = ((x-distance.x)**2 + (y-distance.y)**2)**0.5    
    return distance
# def Get_Dose_Stats(patient, organs, ptvs):
    

def Contour_to_Polygon(contour):
    if len(contour) == 0:
        return None
    contour_xy = [Point(contour[i][0:2]) for i in range(len(contour))]

    poly = Polygon(contour_xy)
    return poly

def Polygon_to_Polygon_Distance(poly1 : Polygon, poly2 : Polygon):
    #this function computes the distance (min) between two polygons
    min_dist = 100000
    for point in poly2.exterior.coords:
        dist = Point_to_Polygon_Distance(Point(point), poly1)    
        if dist < min_dist:
            min_dist = dist
    if min_dist == 100000:
        return None        
    return min_dist       

def Distance_Between_ROIs(roi1, roi2):
    #calculate the minimum distance between two rois, slice by slice
    
    #to speed things up, only check the slice which is at the closest z value
    z_vals_roi2 = []
    for i, contour in enumerate(roi2):
        for s, slice in enumerate(contour):
            if len(slice) > 0:
                z_vals_roi2.append([slice[0][2], [i,s]])       #each element has the z location as well as index in list

    minDistance = 1000000
    for slices in roi1:
        for slice in slices:
            if len(slice) == 0:
                continue
            z_slice = slice[0][2]

            closest_slice_idx = min(z_vals_roi2, key=lambda z:abs(z[0]-z_slice))[1]  #get the closest z value index in z_vals_roi_2

            closest_slice = roi2[closest_slice_idx[0]][closest_slice_idx[1]] 

            slice_polygon = Contour_to_Polygon(slice)
            closest_slice_polygon = Contour_to_Polygon(closest_slice)

            dist = Polygon_to_Polygon_Distance(slice_polygon, closest_slice_polygon)
            dist_z = abs(z_slice - closest_slice[0][2])#also need distance in the z direction
            dist = sqrt(dist**2 + dist_z**2)
            if dist < minDistance:
                minDistance = dist
    return minDistance 

def Overlap_Fraction(roi1, roi2):
    #gets the fraction of roi1 that overlaps with roi2 as a percentage
    intersection_area = 0
    total_area = 0


    z_vals_roi2 = []
    for i, contour in enumerate(roi2):
        for s, slice in enumerate(contour):
            if len(slice) > 0:
                z_vals_roi2.append([slice[0][2], [i,s]])   
                    #each element has the z location as well as index in list

    for slices in roi1:
        for slice in slices:
            if len(slice) == 0:
                continue
            slice_polygon = Contour_to_Polygon(slice)
            total_area += slice_polygon.area

            z_slice = slice[0][2]
            for i in range(len(z_vals_roi2)):
                if abs(int(round(z_slice, 2)*100) - int(round(z_vals_roi2[i][0], 2)*100)) < 2:    #if on the same axial plane
                    #calculate overlap if distance between isnt 0
                    roi2_slice = roi2[z_vals_roi2[i][1][0]][z_vals_roi2[i][1][1]]

                    roi2_slice_polygon = Contour_to_Polygon(roi2_slice)
                    intersection_area += slice_polygon.intersection(roi2_slice_polygon).area
                    break
    overlap_area = round(intersection_area / total_area, 3) * 100         
    return overlap_area   
 


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

        



