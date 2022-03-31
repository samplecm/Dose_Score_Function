import Patient
import Contours
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import math

def skew_normal_integrand(t):
    return math.exp(-t**2/2)

def integral(t, p1, p2, p3)    
