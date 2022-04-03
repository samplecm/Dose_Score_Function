import Patient
import Contours
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import Contours
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")    

import math

def skew_normal_integrand(t):
    return np.exp(-t**2/2)

def skew_normal_integral_first(D, p1, p2): 
    p3 = 0 #for first fit keep p3 = 0
    t_bound = p3 * (D-p1) / p2 
    integral = integrate.quad(skew_normal_integrand, -10, t_bound)[0]
    return (integral / (p2*math.pi)) * np.exp(-(D-p1)**2/(2*p2**2))

def skew_normal_integral_p3(D, p3):
   
    t_bound = p3 * (D-p1) / p2 
    integral = integrate.quad(skew_normal_integrand, -10, t_bound)[0]
    return (integral / (p2*math.pi)) * np.exp(-(D-p1)**2/(2*p2**2))


def skew_normal_integral_all(D, p1, p2, p3):
 
    t_bound = p3 * (D-p1) / p2 
    integral = integrate.quad(skew_normal_integrand, -10, t_bound)[0]
    return (integral / (p2*math.pi)) * np.exp(-(D-p1)**2/(2*p2**2))

def Fit_DVH_Params(D_array, volume_array):
    #first fit with p3 = 0
    integral = np.vectorize(skew_normal_integral_first)    
    popt_1, _ = curve_fit(integral, D_array, volume_array, bounds=([0.,0.01], [100, 100]), p0=[0.01,0.01], maxfev=2000)    #p0=[0.01,0.01,0.01],bounds=([0.,0.01,-10.], [100, 100, 100])
    global p1, p2
    p1 = popt_1[0]
    p2 = popt_1[1]
    # integral = np.vectorize(skew_normal_integral_p3)  
    # p3, _ = curve_fit(integral, D_array, volume_array, bounds=(-1000, 1000), p0=0, maxfev=2000)
    p3 = 0

    integral = np.vectorize(skew_normal_integral_all) 
    
    # plt.plot(D_array, volume_array, color="green", label="actual differential dvh")
    # plt.plot(D_array, integral(D_array, popt_1[0], popt_1[1], p3), color="red", label=f"fitted DVH. Params:{popt_1[0]}, {popt_1[1]}, {p3}")
    # plt.legend()
    # plt.show()
    return [p1, p2, p3]  


    

def Get_DVH(dose_voxels, prescription_dose):
    #returns a list containing a list for each subsegment. Each of these list in itself contains 2 lists: a dose bin array and a relative volume bin array. the dose bins are defined by lower bounds.
    num_bins = 200
    dose_bins = np.linspace(0, 1.2, num=num_bins, endpoint=False)
    dose_bin_size = 1.2 / num_bins
    volumes = np.linspace(0,0, num=num_bins)
    for voxel in dose_voxels:
        volume_bin = min(math.floor(voxel / (1.2*prescription_dose/num_bins)), 199)
        volumes[volume_bin] += 1 / dose_bin_size
    volumes /= len(dose_voxels)
    params = Fit_DVH_Params(dose_bins, volumes)   
    return [dose_bins.tolist(), volumes.tolist()] , params





# def skew_normal_integrand(t):
#     return np.exp(-t**2/2)

# def skew_normal_integral_first(D, p1, p2):
#     dt = 0.01    
#     p3 = 0 #for first fit keep p3 = 0
#     t_bound = p3 * (D-p1) / p2 
#     tt = np.arange(-10, t_bound, dt)
#     integrand = skew_normal_integrand(tt)
#     integral = np.trapz(integrand, dx=dt, axis=-1)
#     return (integral / (p2*math.pi)) * np.exp(-(D-p1)**2/(2*p2**2))

# def skew_normal_integral_p3(D, p3):
#     dt = 0.01    
#     t_bound = p3 * (D-p1) / p2 
#     tt = np.arange(-10, t_bound, dt)
#     integrand = skew_normal_integrand(tt)
#     integral = np.trapz(integrand, dx=dt, axis=-1)
#     return (integral / (p2*math.pi)) * np.exp(-(D-p1)**2/(2*p2**2))


# def skew_normal_integral_all(D, p1, p2, p3):
#     dt = 0.1    
#     t_bound = p3 * (D-p1) / p2 
#     tt = np.arange(-10, t_bound, dt)
#     integrand = skew_normal_integrand(tt)
#     integral = np.trapz(integrand, dx=dt, axis=-1)
#     return (integral / (p2*math.pi)) * np.exp(-(D-p1)**2/(2*p2**2))

# def Fit_DVH_Params(D_array, volume_array):
#     #first fit with p3 = 0
#     integral = np.vectorize(skew_normal_integral_first)    
#     popt_1, _ = curve_fit(integral, D_array, volume_array, bounds=([0.,0.01], [100, 100]), p0=[0.01,0.01], maxfev=2000)    #p0=[0.01,0.01,0.01],bounds=([0.,0.01,-10.], [100, 100, 100])
#     global p1, p2
#     p1 = popt_1[0]
#     p2 = popt_1[1]
#     integral = np.vectorize(skew_normal_integral_p3)  
#     p3, _ = curve_fit(integral, D_array, volume_array, bounds=(-1000, 1000), p0=0, maxfev=2000)

#     integral = np.vectorize(skew_normal_integral_all)   
#     plt.plot(D_array, volume_array, color="green", label="actual differential dvh")
#     plt.plot(D_array, integral(D_array, popt_1[0], popt_1[1], p3), color="red", label=f"fitted DVH. Params:{popt_1[0]}, {popt_1[1]}, {p3[0]}")
#     plt.legend()
#     plt.show()

