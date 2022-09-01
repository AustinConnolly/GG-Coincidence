# Getting the Callibration factor
# 29/8/2021
# Austin COnnolly
# This code will read the csv files for the various elements and then
# I will input the equivalent Energy value to get the callibration for the data

## Importing Libraries ----------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from scipy.optimize import curve_fit
from os import listdir
import os
from os.path import isfile, join
## ------------------------------------------------------------------------------------------------------------------------------

## Reading in the files ---------------------------------------------------------------------------------------------------------
path1 = os.path.dirname(os.path.abspath(__file__)) + '\\PHY3004W_gglab_singles_spectra\\' # This gives the path of the current python to read other files in the same path.
onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))] # Gets the names of the files in the path
o=0
c=0
sources = ['Co60','Cs137','Na22']
Doublesources = ['Co60','Na22']
filenm = onlyfiles
sigmas=[]
mus=[]
redfiles=[]

bkgrnd = []

for i in range(len(filenm)):                                    # Collecting the Red files to perform energy callibration
    if filenm[i].find("Blue")>=0:
        if filenm[i].find("Back")>=0:
            bkgrnd.append(filenm[i])
        else:
            redfiles.append(filenm[i])
print(redfiles)

ends = [[381,438],[438,488],[196,272],[148,216],[408,490]]
guess_mus = [409,464,230,178,443]
guess_ds = [250,135,10,400,30]
guess_As = [1644,1351, 14360,11343,1444]
Backdata = pd.read_csv(path1+bkgrnd[0],header = None, skiprows=6)
h=0
jj=0
p=0

Done = False
while h < 3:

    udata = []
    ydata = []
    tdata = []
    b=0
    nm = path1 + redfiles[h] # path and name of the file

    if redfiles[h].find(Doublesources[jj])>=0 and Done==False:
        h=h-1
        Done = True
        print(h)
    elif redfiles[h].find(Doublesources[jj])>=0 and Done==True:
        Done = False
        jj+=1
        print(h)
    data = pd.read_csv(nm,header = None,skiprows=6).values # Collects the values as a 2-D array

    channel = data[:,0]
    counts = data[:,1]              #- Backdata[:,1]

    plt.plot(channel,counts)
    plt.show()

    N = len(channel)

    while b<N:
        if channel[b]>ends[p][0] and channel[b]<ends[p][1]:
            udata.append(1/np.sqrt(counts[b]))
            ydata.append(counts[b])
            tdata.append(channel[b])
        b=b+1

    def f(x, A, mu, sigma, D):  
        return A*np.exp(-((x-mu)**2)/(2*sigma**2)) + D
    
    A0 = guess_As[p]
    mu0 = guess_mus[p]
    sigma0 = (ends[p][1]-ends[p][0])/(0.2*len(tdata))  
    D0 = guess_ds[p]
    p0 = [A0, mu0, sigma0, D0]
    name = ['A', 'mu', 'sigma', 'Vertical Shift']
    
    print(sigma0)
    print(mu0)
    tmodel = np.linspace(ends[p][0], ends[p][1], 10000)
    
    ystart=f(tmodel,*p0)
    
    ## Curve Fit Function --------------------------------------------------------------------------------------------------------------------------------------
    
    popt,pcov=curve_fit(f,tdata,ydata,p0,sigma=udata,absolute_sigma=True)
    dymin = (ydata-f(tdata,*popt))/udata
    min_chisq = sum(dymin*dymin)
    dof=len(tdata) - len(popt)
    
    
    print('Chi square: ',min_chisq)
    print('Number of degrees of freedom: ',dof)
    print('Chi square per degree of freedom: ',min_chisq/dof)
    print()
    
    mn=min_chisq / dof
    
    print('Fitted parameters with 68% C.I.: ')
    
    for i, pmin in enumerate(popt):
        print('%2i %-10s %12f +/- %10f'%(i, name[i], pmin, np.sqrt(pcov[i,i])*np.sqrt(mn)))
    
    print()
    
    perr=np.sqrt(np.diag(pcov))
    print(perr)
    
    print("Correlation matrix")
    print("               ")
    
    for i in range(len(popt)): print('%-10s'%(name[i],)),
    print()
    
    for i in range(len(popt)):
        print('%10s'%(name[i])),
        for j in range(i+1):
            print('%10f'%(pcov[i,j]/np.sqrt(pcov[i,i]*pcov[j,j]),)),
        print()
    
    mus.append(popt[1])
    sigmas.append(popt[2])
    yfit=f(tmodel,*popt)
    
    ## Plotting ------------------------------------------------------------------------------------------------------------------------------------------------
    plt.scatter(tdata,ydata)
    plt.errorbar(tdata, ydata, xerr = None, yerr = udata, fmt = '', marker='.', ls = 'None',capsize=2.3, ecolor = 'b',label='Data points') # Plots errorbar
    plt.xlabel('Channel') # Labels x-axis
    #plt.plot(tmodel,ystart,'-g') # Plots estimate line
    plt.ylabel('Counts') # Labels y-axis    
    plt.title(redfiles[h])
    plt.plot(tmodel,yfit,'-r', label='Line of best fit') # Plots 
    plt.grid()
    plt.legend()
    plt.tick_params(direction='in',top=True,right=True) # Assigns lines to be on the inside
    
    plt.show()

    print(p)


    p+=1
    h+=1

#channel = energy * gradient

Energy = [1173.228,1332.492,661.657,511.0,1274.537]
print(mus)

N = len(Energy)

udata = []
ydata = []
tdata = []
b=0
u = []
while b<N:
    udata.append(sigmas[b])  # sigmas[b]
    tdata.append(Energy[b])
    ydata.append(mus[b])
    b=b+1


def f1(x, m, c):  
    return np.asarray(m)*x+c

m0 = 0.3
c0 = 11
p0 = [m0, c0]
name = ['m', 'c']


tmodel = np.linspace(400, 1400, 10000)

ystart=f1(tmodel,*p0)

## Curve Fit Function --------------------------------------------------------------------------------------------------------------------------------------

popt,pcov=curve_fit(f1,tdata,ydata,p0,sigma=udata,absolute_sigma=True)
dymin = (ydata-f1(tdata,*popt))/udata
min_chisq = sum(dymin*dymin)
dof=len(tdata) - len(popt)


print('Chi square: ',min_chisq)
print('Number of degrees of freedom: ',dof)
print('Chi square per degree of freedom: ',min_chisq/dof)
print()

mn=min_chisq / dof

print('Fitted parameters with 68% C.I.: ')

for i, pmin in enumerate(popt):
    print('%2i %-10s %12f +/- %10f'%(i, name[i], pmin, np.sqrt(pcov[i,i])*np.sqrt(mn)))

print()

perr=np.sqrt(np.diag(pcov))
print(perr)

print("Correlation matrix")
print("               ")

for i in range(len(popt)): print('%-10s'%(name[i],)),
print()

for i in range(len(popt)):
    print('%10s'%(name[i])),
    for j in range(i+1):
        print('%10f'%(pcov[i,j]/np.sqrt(pcov[i,i]*pcov[j,j]),)),
    print()


yfit=f1(tmodel,*popt)

## Plotting ------------------------------------------------------------------------------------------------------------------------------------------------
plt.scatter(tdata,ydata)
plt.errorbar(tdata, ydata, xerr = None, yerr = udata, fmt = '', marker='.', ls = 'None',capsize=2.3, ecolor = 'b',label='Data points') # Plots errorbar
plt.xlabel('Energy (keV)') # Labels x-axis
plt.ylabel('Channel') # Labels y-axis    
plt.plot(tmodel,yfit,'-r', label='Line of best fit') # Plots 
xmin,xmax,ymin,ymax = plt.axis([400,1400,150,500])
plt.grid()
plt.legend()
plt.tick_params(direction='in',top=True,right=True) # Assigns lines to be on the inside
plt.show()

print(popt[0])