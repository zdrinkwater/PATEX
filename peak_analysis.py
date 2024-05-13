# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:30:16 2023

@author: Zach Drinkwater
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from xrd_module import XRDPeakData

# NOTE: Data was collected at 0.1 degree per reading so 1 idx = 0.01 degree.

lam = 1.5406e-10 # X-ray wavelength
#filename = 'STO sub_2theta-om_15-115deg_2-Theta_Omega.txt'

#filename = 'RH001DS_2theta-om_15-115deg_0.1degmin-1_37-18_2-Theta_Omega.txt' # Could be RH004DW
#filename = 'RH001DS_2theta-om_15-115deg_17hr_2-Theta_Omega.txt'
#filename = 'RH005DW_2theta-om_15-115_deg_37-18_0.1deg_2-Theta_Omega.txt'

filename = 'REAL553_2theta_data.dat'
#FILM: sigma = 3, min = 1e3, max = 2e5, others = 40, 100, 50, [1, 100]
#SUBSTRATE: sigma = 3, min = 3e5, max = 1e10, others = 40, 100, 50, [1, 100]

# DATA FROM GROUP
filename = 'RH014JA_2theta-om_15-115deg_1-1-1_2-Theta_Omega.txt'

def cm_to_idx(cm):
    return cm*100 - 1500

def idx_to_cm(idx):
    return (idx + 1500)/100


min_peak_intensity = 1e1 # Minimum peak intensity
max_peak_intensity = 1e10 # Maximum peak intensity

idx_halfrange = 40 # Index range to fit over (plus/minus) where 100 = 1 degree
distance = 50 # Minimum distance between peaks (idx)
prominence = 10 # Minimum prominence of peaks
width = [1, 100] # Width range of peaks
peak_params = [min_peak_intensity, max_peak_intensity, idx_halfrange, distance, prominence, width]

data1 = XRDPeakData(filename, lam, skiprows=3)
raw = data1.plot_raw_peak_data()

sigma = 3
data1.filter_data(sigma)

# NOTES:
# 1. Use film=False for substrate peaks and film=True for film peaks.
# 2. Use film_left=True if the films peaks are to the left of substrate peaks
#    (indicating larger lattice parameter) or film_left=False if not.
# 3. Currently using FWHM/2 for the error in the peak locations.
peak_data = data1.fit_all_peaks(peak_params, curve='g', film_left=True, film=True, num_peaks=3)

try:
    for i in range(len(peak_data[0, :])):
        if peak_data[0, i] > 0 and peak_data[0, i] < 180 and peak_data[1, i] < 1e10 and peak_data[3, i] < 1 and peak_data[2, i] < 2:
            print(f'Peak at 2theta = {peak_data[0, i]:.2f} ± {peak_data[3, i]:.2f} degrees has intensity {peak_data[1, i]:.4g} ± {peak_data[4, i]:.2g}'
                  + f' and FWHM {peak_data[2, i]:.3f} ± {peak_data[5, i]:.3f} degrees')
    d = data1.lattice_parameter
    d_err = data1.lattice_parameter_err
    print(f'The lattice parameter (d) is {d*1e10:.3f} ± {d_err*1e10:.3f} A')
except TypeError:
    print('Error')
