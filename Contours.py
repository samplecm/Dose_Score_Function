class Contours(object):

    
    def __init__(self, name, dicomName, contours):
        self.wholeROI = contours
        self.roiName = name
        self.dicomName = dicomName
        self.dose_params = None
        self.dose_bins = None
        self.dose_array = None 
