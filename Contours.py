class Contours(object):

    
    def __init__(self, name, dicomName, contours):
        self.wholeROI = contours
        self.roiName = name
        self.dicomName = dicomName
        self.ptv70_dist = None
        self.ptv63_dist = None
        self.ptv56_dist = None
        self.ptv30_dist = None
        self.ptv35_dist = None
        self.ptv40_dist = None
        self.ptv45_dist = None
        self.ptv50_dist = None
        self.ptv54_dist = None
        self.ptv55_dist = None
        self.ptv60_dist = None
        self.dose = None
