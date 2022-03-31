import copy 
import Chopper
import os
import numpy as np
from shapely.geometry import Polygon, Point, LineString, polygon
from math import cos, sin, pi
import pickle 
from Contours import Contours
from Patient import Patient
import statistics 
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

def Interpolate_Slices(oar, ptv, centre_z):
    #add any missing slices from ptv that are in oar and within ptv range. centre_z is the centre of the oar, where a slice in the ptv must also be interpolated

    #first get z vals in oar
    new_ptv = copy.deepcopy(ptv)
    z_vals = []
    for slice in ptv:
        slice = slice[0]
        if len(slice) != 0:
            z_vals.append(slice[0][2])

    z_min = min(z_vals)
    z_max = max(z_vals)

    for slice in oar:
        slice = slice[0]
        if len(slice) != 0:
            z = slice[0][2]
            if z in z_vals:
                continue
            elif z > z_max or z < z_min:
                continue

            #need to interpolate this contour in ptv
            contoursZ = Chopper.ClosestContourZ(z, ptv) #Get the closest contour above and below the cut
            newContour = []
            for c, contour in enumerate(ptv[contoursZ[0]]):
                newContour.append([])
                try:
                    closest_slice = ptv[contoursZ[1]][c]
                except IndexError:
                    continue 
                for j in range(len(contour)): #Go over all points in closest contour
                    point1 = contour[j]      
                    point2 = Chopper.ClosestPoint(point1, closest_slice, island_idx=True)    #Now get the closest point in second closest contour
                    if point2 == None:
                        break
                    newPoint = Chopper.InterpolateXY(point1, point2, z)    #now interpolate between the two
                    #add new point to new contour
                    newContour[-1].append(newPoint)
            #add this new contour to newContoursList
            new_ptv.append(newContour)    
    
    #also add slice at centre point of oar (if needed)
    if centre_z < z_max and centre_z > z_min and centre_z not in z_vals:
        contoursZ = Chopper.ClosestContourZ(centre_z, new_ptv) #Get the closest contour above and below the cut
        newContour = []
        for c, contour in enumerate(new_ptv[contoursZ[0]]):
            newContour.append([])
            try:
                closest_slice = new_ptv[contoursZ[1]][c]
            except IndexError:
                continue 
            if closest_slice == []: 
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
        new_ptv.append(newContour)    

    


    new_ptv.sort(key=Chopper.GetZVal)     
    return new_ptv 


def AddInterpolatedPoints(orig_contours):
    """This makes sure that each slice of contours has at least 100 points
    Args: 
        contours (list): the contour list for a single patient
    """
    contours = copy.deepcopy(orig_contours)
    #Now add in sufficient number of points 
    for contour_idx in range(len(contours)): 
        contour = contours[contour_idx]
        numPointsOrig = len(contour)
        numPoints = len(contour)
        if numPoints > 100:
            continue
        if numPoints < 4:
            contours[contour_idx] = []
            continue

        pointIncreaseFactor = 1
        while numPoints < 100:  
            numPoints = numPoints + numPointsOrig
            pointIncreaseFactor = pointIncreaseFactor + 1      
        increasedContour = []
        for point_idx in range(len(contour)-1):    
            increasedContour.append(contour[point_idx].copy())
            for extraPointNum in range(pointIncreaseFactor):
                scaleFactor = extraPointNum / (pointIncreaseFactor + 1)
                newX = (contour[point_idx+1][0] - contour[point_idx][0]) * scaleFactor + contour[point_idx][0]
                newY = (contour[point_idx+1][1] - contour[point_idx][1]) * scaleFactor + contour[point_idx][1]
                z = contour[point_idx][2]
                newPoint = [newX, newY, z]
                increasedContour.append(newPoint)
        #Now do it for the last point connected to the first
        for extraPointNum in range(pointIncreaseFactor):
            scaleFactor = extraPointNum / (pointIncreaseFactor + 1)
            newX = (contour[0][0] - contour[-1][0]) * scaleFactor + contour[-1][0]
            newY = (contour[0][1] - contour[-1][1]) * scaleFactor + contour[-1][1]
            z = contour[-1][2]
            newPoint = [newX, newY, z]
            increasedContour.append(newPoint)        
        contours[contour_idx] = increasedContour

    return contours 