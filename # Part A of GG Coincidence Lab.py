# Part A of GG Coincidence Lab
# 15/9/2021
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
path1 = os.path.dirname(os.path.abspath(__file__)) + '\\PHY3004W_gglab_coinc_spectra\\' # This gives the path of the current python to read other files in the same path.
onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))] # Gets the names of the files in the path
o=0
c=0
filenm = onlyfiles

timenorm = [2600,2092,5275] #[197,157,6650]#

Cofiles=[]
Nafiles = []
bkgrnd = []

for i in range(len(filenm)):                                    # Collecting the Red files to perform energy callibration
    if filenm[i].find("Na")>=0:
        Nafiles.append(filenm[i])
    elif filenm[i].find("Co")>=0:
        Cofiles.append(filenm[i])
print(Cofiles)
print(Nafiles)

Lables = ['Blue detector with a wide gating','Blue detector with gating of 511 keV','Red detector coincidence gating of 511 keV']
calfact = [0.349,0.349,0.349]

for i in range(3):
    nm = path1 + Cofiles[i] # Cofiles[i]

    data = pd.read_csv(nm,header = None,skiprows=6).values # Collects the values as a 2-D array

    channel = (data[:,0]) * (1/(calfact[i]))
    counts = data[:,1] / timenorm[i]  
    plt.plot(channel,counts, label = Lables[i])

plt.xlabel('Energy (keV)') # Labels x-axis
plt.ylabel('Counts per second') # Labels y-axis
xmin,xmax,ymin,ymax = plt.axis([0,2800,0,17.5])
plt.grid()
plt.tick_params(direction='in',top=True,right=True) # Assigns lines to be on the inside
plt.legend()
plt.show()