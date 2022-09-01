# Part A of GG Coincidence Lab
# 16/9/2021
# Austin Connolly
# 

## Importing Libraries ----------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import count
import pandas as pd
from matplotlib import rc
from scipy.optimize import curve_fit
from os import listdir
import os
from os.path import isfile, join
## ------------------------------------------------------------------------------------------------------------------------------

## Reading in the files ---------------------------------------------------------------------------------------------------------
path1 = os.path.dirname(os.path.abspath(__file__)) + '\\PHY3004W_gglab_angular_scans\\' # This gives the path of the current python to read other files in the same path.
onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))] # Gets the names of the files in the path
# o=0
# c=0
# filenm = onlyfiles

# Cofiles=[]
# Nafiles = []
# bkgrnd = []

# for i in range(len(filenm)):                                    # Collecting the Red files to perform energy callibration
#     if filenm[i].find("Scan1")>=0:
#         Nafiles.append(filenm[i])
#     elif filenm[i].find("Scan2")>=0:
#         Cofiles.append(filenm[i])
# print(Cofiles)
# print(Nafiles)
Names = ["TSCA wide open $d_1=d_2=15cm$","511 keV gate $d_1=d_2=15cm$","TSCA wide open with $d_1=15cm;d_2=30cm$"]
for i in range(3):
    nm = path1 + onlyfiles[i]#Cofiles[i]

    data = pd.read_csv(nm,header = None,skiprows=8).values # Collects the values as a 2-D array

    theta = data[:,0]
    N1 = data[:,1] / 30
    N2 = data[:,2] / 30
    Nco = data[:,3] / 30

    plt.plot(theta,Nco, label = Names[i])
plt.xlabel("$\\theta$") # Labels x-axis
plt.ylabel('Counts per second') # Labels y-axis
xmin,xmax,ymin,ymax = plt.axis([-90,90,0,40])
plt.grid()
plt.tick_params(direction='in',top=True,right=True) # Assigns lines to be on the inside
plt.legend()
plt.show()
