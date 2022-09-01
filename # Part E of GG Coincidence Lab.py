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

d1 = [3,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
N1 = [37164,28173,18879,12999,9387,7635,6342,5421,4929,4449,3963,3879,3582,3456,3402,3297,3204,3117,3066,3039]

plt.plot(d1,N1)
plt.show()