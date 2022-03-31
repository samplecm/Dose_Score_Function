from math import nan


class Patient():
    def __init__(self, patientName, patientPath):
        self.path = ""
        self.name = patientName
        self.dose_array = None
        self.brain = None
        self.brainstem = None
        self.brachial_plexus = None
        self.chiasm = None
        self.cochlea = None
        self.pharyngeal_constrictors = None
        self.esophagus = None
        self.globes = None
        self.lacrimal_glands = None
        self.larynx = None
        self.lens = None
        self.lips = None
        self.mandible = None
        self.optic_nerves = None
        self.oral_cavity = None
        self.parotid_left = None
        self.parotid_right = None
        self.spinal_cord = None
        self.submandibular_right = None
        self.submandibular_left = None
        self.thyroid = None
        self.retina = None
        self.dicom_structures = []
        self.PTVs = {}
    
        # self.ptv70 = None
        # self.ptv63 = None
        # self.ptv56 = None
        # self.ptv54 = None
        # self.ptv55 = None
        # self.ptv30 = None
        # self.ptv35 = None
        # self.ptv40 = None
        # self.ptv45 = None
        # self.ptv50 = None
        # self.ptv60 = None



        