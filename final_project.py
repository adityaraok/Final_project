# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:56:18 2016

@author: aditya
"""
import numpy as np
from scipy.optimize import *
import scipy.stats as stats 
import matplotlib.pyplot as plt


# TO SOLVE THE DIFFONENTIAL EQUATION
#

y=[] #a's values
z=[] #c's values
name="Emma"
gender="F"
yd=readData(name,gender)
ac=solver(yd)

for i in range(0,135):
    #print(ac[i][0],ac[i][1])
    y.append(ac[i][0])
    z.append(ac[i][1])

    

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
def plot_diffonential():
    diff=[]
    global y
    global z
    da=holts_model(y)
    dc=holts_model(z)
    #t2=range(1880,2015)
    for t in range(0,135):
        x=np.exp(-da[t]*(t+1))
        y=np.exp(-dc[t]*(t+1))
        avg=average(yd)
        diff.append(avg*(x-y))
    #print(diff)
    return diff




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


    

def plot():
    global name
    plt.figure()
    plt.plot(t1, y,label="co-efficient a")
    plt.xlabel('$x$')
    plt.ylabel("Co-efficient")
    plt.plot(t1, z,label="co-efficient c")
    #plt.plot(t1,itr_gaussian_estimator_c(), label="iterative gaussian estimate of c")
    #plt.plot(t1,gaussian_estimator_c(), label="gaussian estimate of c")
    #plt.plot(t1,gaussian_estimator_a(), label="gaussian estimate of a")
    #plt.plot(t1,itr_gaussian_estimator_a(), label="iterative gaussian estimate of a")
    plt.plot(t1,holts_model(z),label="Holt's model estimate of c")
    plt.plot(t1,holts_model(y),label="Holt's model estimate of a")
    diff=plot_diffonential()
    plt.figure()
    plt.plot(t1,diff, label="Estimated name plot for "+name)
    plt.legend(loc='upper left')
    plt.show()
    
##################################################################  
def itr_gaussian_estimator_c():
    a=[None]*16
    b=[None]*16
    G1=[[],[],[],[],[],[],[],[],[]]
    G=[]
    for j in range(0,8):
        c_maxima=-99
        c_maxima_pos=0
        low=17*(j)
        high=17*(j+1)
        if(high>135):
            high=135
        for i in range(low,high):
            if(ac[i][1]>c_maxima):
                c_maxima=ac[i][1]
                c_maxima_pos=i
    #c's estimate
        gaussian_height=c_maxima
        mu=c_maxima_pos
        sigma=0.02/gaussian_height
        a[j]=0.05/(np.sqrt(2*np.pi)*sigma) 
        for t in range(0,135):    
            b[j]= np.exp(-(t-mu)*(t-mu)/(2*sigma*sigma))
            G1[j].append(a[j]*b[j])
        print(c_maxima,c_maxima_pos)
    for col in range(0,135):
        gsum=0
        for row in range(0,8):
            gsum=gsum+G1[row][col]
        G.append(gsum)
    return G
    
def itr_gaussian_estimator_a():
    a=[None]*8
    b=[None]*8
    G1=[[],[],[],[],[],[],[],[]]
    G=[]
    for j in range(0,4):
        a_minima=99
        a_minima_pos=0
        low=34*(j)
        high=34*(j+1)
        if(high>135):
            high=135
        for i in range(low,high):
            if(ac[i][0]<a_minima):
                a_minima=ac[i][1]
                a_minima_pos=i
    #c's estimate
        gaussian_height=a_minima
        mu=a_minima_pos
        sigma=0.04/gaussian_height
        a[j]=0.25/(np.sqrt(2*np.pi)*sigma) 
        for t in range(0,135):    
            b[j]= np.exp(-(t-mu)*(t-mu)/(2*sigma*sigma))
            G1[j].append(a[j]*b[j])
        print(a_minima,a_minima_pos)
    for col in range(0,135):
        gsum=0
        for row in range(0,4):
            gsum=gsum-G1[row][col]
        G.append(gsum)
    return G
 
#########################################################
def gaussian_estimator_c():
    c_maxima=-99
    c_maxima_pos=0
    G=[]
    for i in range(0,135):
        if(ac[i][1]>c_maxima):
            c_maxima=ac[i][1]
            c_maxima_pos=i
    #c's estimate
    gaussian_height=c_maxima
    mu=c_maxima_pos
    sigma=0.4/gaussian_height
    a=1.0/(np.sqrt(2*np.pi)*sigma)
    for t in range(0,135):    
        b= np.exp(-(t-mu)*(t-mu)/(2*sigma*sigma))
        G.append(a*b)
    print(c_maxima,c_maxima_pos)
    return G
    
def gaussian_estimator_a():
    a_minima=99
    a_minima_pos=0 
    G=[]
    for i in range(1,135):
        if(ac[i][0]<a_minima):
            a_minima=ac[i][0]
            a_minima_pos=i
    gaussian_height=a_minima
    mu=a_minima_pos
    sigma=0.2/gaussian_height
    a=0.5/(np.sqrt(2*np.pi)*sigma)
    for t in range(0,135):    
        b= np.exp(-(t-mu)*(t-mu)/(2*sigma*sigma))
        G.append(a*b)
    print(a_minima,a_minima_pos)
    return G
############################################################################
#def diff_func(P,Q):
#    avg=Q[0]
#    peak_val=Q[1]
#    peak_pos=Q[2]
#    a=P[0]
#    c=P[1]
#    x=(np.log(c/a) / (c-a)) - peak_pos
#    y=avg*(np.exp(-a*peak_pos)-np.exp(-c*peak_pos)) - peak_val
#    print(peak_pos,peak_val)
#    #z=(np.log(c/a) / (c-a)) - peak_pos
#    return [x,y]
#
#def diff_estimator_c():
#    guess=(2,3)
#    y=[]
#    sum=0
#    c_maxima=-99
#    c_maxima_pos=0
#    for i in range(0,135):
#        if(ac[i][1]>c_maxima):
#            c_maxima=ac[i][1]
#            c_maxima_pos=i
#    for d in yd:
#        sum=sum+d
#    avg=sum/len(yd)
#    peak_val=c_maxima
#    peak_pos=c_maxima_pos
#    [a,c]=fsolve(diff_func,guess,[avg,peak_val,peak_pos])
#    print(a,c)
#    for t in range(1,136):
#        y.append((np.exp(-a*t)- np.exp(-c*t)))
#    return y
    
##############################################################################
#def gamma_dist(P,Q):
#    peak_val=Q[0]
#    peak_pos=Q[1]
#    k=P[0]
#    theta=P[1]
#    #x=k*theta-peak_pos
#    x=(k*theta)- (1/peak_val)
#    #y=k*theta-peak_pos
#    y=(k*theta)- (1/peak_val)
#    print(peak_pos,peak_val)
#    #z=(np.log(c/a) / (c-a)) - peak_pos
#    return [x,y]
#
#def gamma_estimator_c():
#    guess=(0,0)
#    y=[]
#    c_maxima=-99
#    c_maxima_pos=0
#    for i in range(0,135):
#        if(ac[i][1]>c_maxima):
#            c_maxima=ac[i][1]
#            c_maxima_pos=i
#    peak_val=c_maxima
#    peak_pos=c_maxima_pos
#    [k,theta]=fsolve(gamma_dist,guess,[peak_val,peak_pos])
#    print(k,theta)
#    gamma=math.gamma(k)
#    for t in range(1,136):
#            num = np.power(t,k-1)*np.exp(t/theta)    
#            den = gamma*np.power(theta,k)
#            y.append(num/den)
#    #return y
#    return np.random.gamma(k,theta,size=135)
##############################################################################

def gamma_fit(data):
    fit_alpha, fit_loc, fit_beta=stats.gamma.fit(data)
    print(fit_alpha, fit_loc, fit_beta)
    #return np.random.gamma(fit_alpha,fit_beta,size=135)
    return stats.gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=135)

      
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

#############################################################################
def gamma_func(x,k,theta):
    num = (np.power(x,k-1)*np.exp(x/theta) ) / (math.gamma(k)*np.power(theta,k)  )
    return num
    
def diff_func(x,a,b,c):
    return b*(np.exp(-a*x) - np.exp(-c*x))
#############################################################################

#x1=np.linspace(0,135,135)
#print(curve_fit(gamma_func,x1,y))

def holts_model(yearlyData):
    arr_len=len(yearlyData)
    if arr_len!=0:
        a=[None]*arr_len
        b=[None]*arr_len
        alpha=0.9
        beta=0.09
        #print(yearlyData)
        a[0]=yearlyData[0]
        b[0]=(yearlyData[len(yearlyData)-1]-yearlyData[0])/len(yearlyData)
        forecast=[None]*arr_len
        forecast[0]=yearlyData[0]
        for t in range(1,len(yearlyData)):
            a[t]=alpha*(float(yearlyData[t])) + (1-alpha)*(float(a[t-1]+b[t-1]))
            b[t]=beta*(float)(a[t]-a[t-1]) + (1-beta)*(float)(b[t-1])
            forecast[t]=a[t-1]+b[t-1]
        #print forecast
        return forecast
######################################################################









t1=range(1880,2015)
plot()

