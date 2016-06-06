# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:56:18 2016

@author: aditya
"""
import numpy as np
from scipy.optimize import *

import matplotlib.pyplot as plt


# TO SOLVE THE DIFFONENTIAL EQUATION
#

def diffonential_function(A,B):
    a=A[0]
    c=A[1]
    k=B[0]
    avg=B[1]
    t=B[2]
    x=avg*(np.exp(-a*t))
    y=avg*(np.exp(-c*t))
    F=[None,None]
    F[0]=x-y-k
    F[1]=x-y-k
    return F
    
def solver(data):    
    guess  = (0,0)
    sum=0
    ac=[]
    time=0
    for datum in data: 
        sum = sum+datum
    avg=sum/len(data)
    for datum in data:
        time=time+1
        print(datum)
        z=fsolve(diffonential_function,guess,[datum,avg,time])
        ac.append(z)
    print(avg)
    return ac






##############################################################################
def parameter_distribution():
    n=135
    total=0
    p=0
    for val in yd:
        total=total+val
    for val in yd:
        p=p+(val/total)
    p=p/n
    print(p)





##############################################################################














# THIS FUNCTION READS DATA FROM THE FILES 
#



female_names=set([])
male_names=set([])

def readData(babyName,babyGender):
    #name=raw_input("Baby name ? ")
    #gender=raw_input("Gender ? ")
    name=babyName
    gender=babyGender
    #upper_bound=input("Prediction for year ? ")
    upper_bound=2015
    total=0
    yearlyData=[]
    #partial_total=0
    for i in range(1880,upper_bound):
        f=open("/home/aditya/Documents/Final Project UCLA/names/yob"+str(i)+".txt",'r')
        name_found_flag=0
        for line in f:
            line_arr=line.split(',')
            if line_arr[1]=="F":
                female_names.add(line_arr[0])
            else:
                male_names.add(line_arr[0])
            if(line_arr[0]==name and line_arr[1]==gender):
                yearlyData.append(int(line_arr[2]))
                #print line,i
                name_found_flag=1
                total=total+int(line_arr[2])
        f.close()
        if (name_found_flag==0):
            yearlyData.append(0) 
    return yearlyData

y=[] #a's values
z=[] #c's values
yd=readData("Mike","M")
ac=solver(yd)

for i in range(0,135):
    print(ac[i][0],ac[i][1])
    y.append(ac[i][0])
    z.append(ac[i][1])
    

def plot(x,y):
    plt.figure()
    plt.plot(x, y,label="co-efficient a")
    plt.xlabel('$x$')
    plt.ylabel("Co-efficient")
    plt.plot(x, z,label="co-efficient c")
    plt.legend()
    plt.show()
    
    
##############################################################################    
#t = np.asarray(range(1880, 2015))
#data=np.asarray(z)
#guess_mean =np.mean(data)
#guess_std = 3*np.std(data)/(2**0.5)
#guess_phase = 0
#
## we'll use this to plot our first estimate. This might already be good enough for you
#data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean
#
## Define the function to optimize, in this case, we want to minimize the difference
## between the actual data and our "guessed" parameters
#optimize_func = lambda x: x[0]*np.sin(t+x[1]) + x[2] - data
#est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]
#
## recreate the fitted curve using the optimized parameters
#data_fit = est_std*np.sin(t+est_phase) + est_mean
#
#plt.plot(data, '.')
#plt.plot(data_fit, label='after fitting')
#plt.legend()
#plt.show()
##############################################################################

x=range(0,135)
plot(x,y)