#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 01:45:32 2021

@author: joe
"""

#from matplotlib import pyplot as plt  
from math import * 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import simps


#v = M**4*(1-sym.exp(-np.sqrt(2./3)*phi))**2


#All analytic!
fNL = .01325
gNL = 1.446*10**-5
nS = .9678
alphaS = -.0043815
logM = np.linspace(-7,-5,100000)#(-8,-3,10)
#^ NEED at lest 10000! If 100, off by a factor of at least 2

#As = M**4 * 22.26
#logAs = [log(22.26) + log(exp(logM[i])**4) for i in range(len(logM))]
logAs = [log(22.26) + 4*logM[i] for i in range(len(logM))]

#samples2 = np.loadtxt('/Users/joe/Documents/Research/COM_CosmoParams_fullGrid_R3.01/base_nrun/plikHM_TTTEEE_lowl_lowE/dist/base_nrun_plikHM_TTTEEE_lowl_lowE.covmat')
primordial_covariance = np.array([[2.8851542e-04,-3.541251e-06,-3.2953701e-05 ,0,0],
                                  [-3.541251e-06,2.1380625e-05,9.9529911e-06,0,0],
                                  [-3.2953701e-05,9.9529911e-06,4.5186311e-05,0,0],
                                  [0,0,0,5.1**2,0],#.8**2,0],#5.1**2,0],
                                  [0,0,0,0,(6.5e4)**2]])#np.array([[samples[4][4],samples[4][5]],[samples[5][4],samples[5][5]]])
#eg 12.9 https://wiki.cosmos.esa.int/planck-legacy-archive/images/4/43/Baseline_params_table_2018_68pc_v2.pdf in says 0.0046, which squared is about 2e-5. This is for 2018 data, but clsoe enough

inverse_primordial_covariance = np.linalg.inv(primordial_covariance)
log10_10As = 3.049
ns = 0.9635
alpha = -0.0055
fNL_bestfit = -.9 
gNL_bestfit = -5.8e4
kval = 1e-30
likelihood = np.zeros([len(logM)])
logAsth = np.zeros([len(logM)])
for i in range(len(logM)):
    ns_th = nS ; alpha_th = alphaS                                                    
    fNL_th = fNL;gNL_th = gNL
    log10_10As_th = log(1e10*exp(logAs[i]))
    logAsth[i] = log10_10As_th
    yd_minus_yth = [log10_10As - log10_10As_th,ns-ns_th,alpha - alpha_th,fNL_bestfit-fNL_th,gNL_bestfit-gNL_th]
    this = np.dot(inverse_primordial_covariance,yd_minus_yth)
    chisq =  np.dot(np.transpose(yd_minus_yth),this)  # is it < dof?
    likelihood[i]= exp(-.5*chisq)
import scipy.integrate as integ
plt.figure()
plt.plot(logM, likelihood)
plt.title('Likelihood, where logM = '+str(logM[0])+' to '+str(logM[-1])+', with '+str(len(logM))+' points.')
plt.show()

from scipy.integrate import cumtrapz
integral = cumtrapz(likelihood,logM)[-1] #Integral = 0.007369

#evidence = 0.007369 / prior

#Prior
prior = np.zeros([len(logM)])
inside = 0
total = 0
for i in range(len(logM)):
    # BE CAREFUL DO WE NEED nS IN AS??????
    log10_10As_th = log(1e10*exp(logAs[i]))
    if log10_10As_th >= 2.85 and log10_10As_th <= 3.25:
        prior[i] = 1
        inside += 1
        total += 1
    else:
        prior[i] = 0
        total += 1
print('integral = '+str(integral))
print('R = '+str(inside/total))
print('E = '+str(integral / (inside/total)))
#Predictivity
fNL_arr = np.array([-36,-18.5,-.9,16.5,34])
As_arr = np.array([2.93,2.99,3.049,3.109,3.169])

evidences = np.array([ [1.3e-12 ,1.3e-12 ,1.3e-12 ,1.3e-12 ,1.3e-12] , #f = -36
                      [.00012,.00012,.00012,.00012,.00012], #f = -18.5
                      [.086,.086,.086,.086,.086], #f = BF
                      [.00047,.00047,.00047,.00047,.00047], #f = 16.5
                      [1.997e-11,1.997e-11,1.997e-11,1.997e-11,1.997e-11] #f = 34
                      ])
####PLOT#######
X1,X2 = np.meshgrid(As_arr,fNL_arr,indexing='ij') #Args should be reverse order of array
Zn = interpn((As_arr,fNL_arr),evidences,(X1,X2))
fig = plt.figure();cmap = plt.get_cmap('PiYG')
plt.contourf(fNL_arr, As_arr, np.transpose(Zn[:,:]),100,cmap = cmap)
cbar = plt.colorbar()
cbar.set_label('Evidence', rotation=360)
plt.show()
#####END PLOT #####


valsSpline = np.array([ 1.3e-12 ,1.3e-12 ,1.3e-12 ,1.3e-12 ,1.3e-12 , #f = -36
                      .00012,.00012,.00012,.00012,.00012, #f = -18.5
                      .086,.086,.086,.086,.086, #f = BF
                      .00047,.00047,.00047,.00047,.00047, #f = 16.5
                      1.997e-11,1.997e-11,1.997e-11,1.997e-11,1.997e-11 #f = 34
                      ])
Ebar = np.max(valsSpline) / e
# Spline the evidence function, then get Heaviside of that 
x1 = np.array([-36,-18.5,-.9,16.5,34,-36,-18.5,-.9,16.5,34,-36,-18.5,-.9,16.5,34,-36,-18.5,-.9,16.5,34,-36,-18.5,-.9,16.5,34 ])
x2 = np.array([2.93,2.93,2.93,2.93,2.93,2.99,2.99,2.99,2.99,2.99,3.049,3.049,3.049,3.049,3.049,3.109,3.109,3.109,3.109,3.109,3.169,3.169,3.169,3.169,3.169])
pts = list(zip(x1,x2)) 
N = 10   
def interpolate(input_arr, pts=pts, integrate = False, N = N):                
    interp = LinearNDInterpolator(pts,input_arr)
    print('interp is '+str(np.shape(interp)))
    print('pts is '+str(np.shape(pts))) #25, 2
    print('input_arr is '+str(np.shape(input_arr))) #25
    
    x1 = np.linspace(fNL_arr[0],fNL_arr[-1],N)
    x2 = np.linspace(As_arr[0],As_arr[-1],N)                       
    factor = [N/(x1[-1] - x1[0]) ,N/(x2[-1] - x2[0])]
    v = interp(x1[:,None],x2[None,:])
    print('v is '+str(np.shape(v))) #100, 100
    a = np.mgrid[x1[0]*factor[0]:x1[-1]*factor[0], x2[0]*factor[1]:x2[-1]*factor[1]] 
    a = np.rollaxis(a, 0, 3)         # Make 0th axis into the last axis
    a[:,:,0] = a[:,:,0] / factor[0]
    a[:,:,1] = a[:,:,1] / factor[1]
    M = np.shape(a)[1]
    O = np.shape(a)[0]
    a = a.reshape((M * O , 2))   # Now reshape while preserving order

    interp_result = interpn((x1, x2), v, a)
    print('interp_result is '+str(np.shape(interp_result))) #100
    test2 = interp_result.reshape(x1.size, x2.size)
    print('test2 is '+str(np.shape(test2)))
    
    if integrate == True:
        fsmall = np.linspace(fNL_arr[0], fNL_arr[-1], O)
        Asmall = np.linspace(As_arr[0], As_arr[-1], M)    
        #test2 = interp_v_new.reshape(fsmall.size, Asmall.size)
        res = simps(simps(test2, Asmall), fsmall)
        return res
    else:
        return test2#interp_result
print('Starting first interp')
interp_v = interpolate(valsSpline, integrate = False)
print('Done with first interp')
#Now, get H(Ebar - E(f,A))
def H(Ebar,E, fNL,As):
    #E = interp_v(fNL,As) #Zn(fNL,As)
    if E >= Ebar:
        result = 1
    else:
        result = 0
    return result
fH = np.linspace(fNL_arr[0],fNL_arr[-1],N)    
AH = np.linspace(As_arr[0],As_arr[-1],N) 
Heav_E = np.zeros([N,N])

for i in range(N): 
  for j in range(N):
      Heav_E[i][j] = H(Ebar, interp_v[i][j],fH[i], AH[j]) 
Heav_E = Heav_E.reshape(N*N) #reshape (N*N,)        
print('Starting second interp')
fH_2 = np.zeros([N*N])
AH_2 = np.zeros([N*N])
for i in range(N):
    for j in range(N):
        fH_2[i*N + j] = fH[j]
        AH_2[i*N + j] = AH[i]
newpts = list(zip(fH_2,AH_2))# (N*N, 2)
res = interpolate(Heav_E,pts = newpts, integrate = True)                   
print('Done with second interp')

Pred = 1 - res/((fNL_arr[-1] - fNL_arr[0]) * (As_arr[-1]- As_arr[0]))


