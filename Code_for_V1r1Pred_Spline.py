#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:48:56 2021

@author: joe
"""

import numpy as np
from math import *
from matplotlib import tri
from trispec_nonmin_Bessel import output_bessel
from scipy.interpolate import interpn
from scipy.interpolate import griddata
from scipy.integrate import simps
from scipy.interpolate import LinearNDInterpolator
from matplotlib import tri
from matplotlib import pyplot as plt  
from scipy.interpolate import UnivariateSpline
import time
import scipy.integrate as integ

# -------------------------------------
primordial_covariance = np.array([[2.8851542e-04,-3.541251e-06,-3.2953701e-05 ,0,0],
                                  [-3.541251e-06,2.1380625e-05,9.9529911e-06,0,0],
                                  [-3.2953701e-05,9.9529911e-06,4.5186311e-05,0,0],
                                  [0,0,0,5.1**2,0],#5.1**2,0],
                                  [0,0,0,0,(6.5e4)**2]])#np.array([[samples[4][4],samples[4][5]],[samples[5][4],samples[5][5]]])
#eg 12.9 https://wiki.cosmos.esa.int/planck-legacy-archive/images/4/43/Baseline_params_table_2018_68pc_v2.pdf in says 0.0046, which squared is about 2e-5. This is for 2018 data, but clsoe enough
print('sigmafNL = '+str(sqrt(primordial_covariance[3,3])))
print('sigmagNL = '+str(sqrt(primordial_covariance[4,4])))
inverse_primordial_covariance = np.linalg.inv(primordial_covariance)
ns = 0.9635
log10_10As = 3.049
alpha = -0.0055
fNL_bestfit = -.9 #2018
print('fNL_BF = '+str(fNL_bestfit))
gNL_bestfit = -5.8e4
print('gNL_BF = '+str(gNL_bestfit))
kval = 1e-30
Likemax = 0 #Keeping track of max Like found so far
Maxloc = [0,0,0,0]
bessel_Bool = True
bessel_r = 1
#######################################





begin = time.time()


def ekphase(n,b,N = 60):
    eps_ek = 3*(N+1)**n
    epsN = n*eps_ek/(N+1)
    epsNN = n*(n-1)*eps_ek/(N+1)**2
    c = sqrt(2*eps_ek) 
    nS = 1.-2*eps_ek/(eps_ek - 1.)*(b/c - 1.) - 7./3*epsN/eps_ek
    alphaS = -b/c*epsN/eps_ek + 7./3 * (epsNN/eps_ek - (epsN/eps_ek)**2) 
    return [nS,alphaS,c]

def get_b(n,N=60):
    epsek = 3*(N+1)**n
    return sqrt(2*epsek)*(1+(ns -1+7/3*n/(N+1))/(-2*epsek/(epsek-1)))



start = 8
size = 256
dim = 2*size
c = np.linspace(0,5.5,dim)
d = np.linspace(0,.08,dim)
lnv = np.linspace(-25,-15,dim)
n = np.linspace(0,1,dim)

summation = 0
numpoints = 0
L_lnL = 0
Likemax = 0

nhere = [n[int(size/start + i*dim/start)] for i in range(start)]
lnvhere = [lnv[int(size/start + i*dim/start)] for i in range(start)]
dhere = [d[int(size/start + i*dim/start)] for i in range(start)]
chere = [c[int(size/start + i*dim/start)] for i in range(start)]
likelihood = np.zeros([start,start,start,start])
fNLarr = np.zeros([start,start,start,start])
gNLarr = np.zeros([start,start,start,start])

kfirst = False
ilast = False
jlast = False
klast = False
llast = False
lfirst = False
for i in range(len(dhere)):
    print(i)
    print('So far, max like is '+str(np.max(likelihood)))
    for j in range(len(nhere)): 
        #print('j = '+str(j))
        bhere = get_b(nhere[j])
        ns_th = ns; alphas_th = ekphase(nhere[j],bhere)[1]
        c_ek = sqrt(2* 3*(60+1)**nhere[j])
        for k in range(len(lnvhere)):
           # print('k = '+str(k))
            for l in range(len(chere)):
                #print('l = '+str(l))
                fNL_th,gNL_th,output_logAs,tau = output_bessel(bhere,dhere[i],chere[l],lnvhere[k],bessel_r,bessel_Bool)


                log10_10As_th = log(1e10*exp(output_logAs+ log((tau*kval/2)**(ns-1))))#log((kval*tau/2)**(-2*c_ek*(bhere-c_ek)/(c_ek**2-2)))))

                yd_minus_yth = [log10_10As - log10_10As_th,ns-ns_th,alpha-alphas_th,fNL_bestfit-fNL_th,gNL_bestfit-gNL_th]
                this = np.dot(inverse_primordial_covariance,yd_minus_yth)
                chisq =  np.dot(np.transpose(yd_minus_yth),this)  # is it < dof? 

                likelihood[i][j][k][l]= exp(-.5*chisq)
                fNLarr[i][j][k][l]= fNL_th
                gNLarr[i][j][k][l]= gNL_th
                if exp(-.5*chisq) > .001:
                    if k == 0 and kfirst==False:
                        print('Problem? k = '+str(k))
                        kfirst = True
                    if l == 0 and lfirst==False:
                        print('Problem? l = '+str(l))
                        lfirst = True
                    if i == start-1 and ilast==False:
                        print('Problem? i = '+str(i))
                        ifound = True
                    if j == start-1 and jlast==False:
                        print('Problem? j = '+str(j))
                        jfound = True
                    if k == start-1 and klast==False:
                        print('Problem? k = '+str(k))
                        kfound = True
                    if l == start-1 and llast==False:
                        print('Problem? l = '+str(l))
                        lfound = True
                    

                summation +=exp(-.5*chisq)
                L_lnL += exp(-.5*chisq)*(-.5*chisq)
                numpoints += 1
                
                        
                if exp(-.5*chisq) > Likemax:
                    Likemax = exp(-.5*chisq)
                    Maxloc = [dhere[i],nhere[j],lnvhere[k],chere[l]]

like_prev = likelihood
threshold = 1e-8
summation_else = 0
#summation = 0.010574274607809233
#numpoints = 4096
Eprime = summation/numpoints 
Log_new = L_lnL/numpoints
print('After, s over n (which is Eprime) is '+str(Eprime))
print('And L_lnL /N = '+str(Log_new))

for q in range(2): 
    begin = time.time()
    lnvr = [lnv[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))] 
    nr = [n[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))]
    dr = [d[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))]
    cr = [c[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))]

    likelihoodr = np.zeros([len(dr),len(nr),len(lnvr),len(cr)])
    fNLr = np.zeros([len(dr),len(nr),len(lnvr),len(cr)])
    gNLr = np.zeros([len(dr),len(nr),len(lnvr),len(cr)])
    Eprime = 0
    Log_new = 0
    
    ilast = False
    llast = False
    klast = False
    mlast = False
    ifirst = False
    mfirst = False
    kfirst = False
   
    
    nhere = [n[int(size/start/(2**q)*(1 + 2*i))] for i in range(int(2**(3+q)))]
    chere = [c[int(size/start/(2**q)*(1 + 2*i))] for i in range(int(2**(3+q)))]
    lnvhere = [lnv[int(size/start/(2**q)*(1 + 2*i))] for i in range(int(2**(3+q)))] 
    dhere =[d[int(size/start/(2**q)*(1 + 2*i))] for i in range(int(2**(3+q)))]
    

    for i in range(int(2**q*start)):
        if i%10 == 0:
            print('At i = '+str(i)+' , max like_ref is '+str(np.max(likelihoodr)))

        for l in range(int(2**q*start)):

            ns_th = ns
            bhere = get_b(nhere[l])
            

            ns_th = ekphase(nhere[l],bhere)[0]
            alphas_th = ekphase(nhere[l],bhere)[1]
            for k in range(int(2**q*start)):
                for m in range(int(2**q*start)):


                    if like_prev[i][l][k][m] >= threshold:
                        fNL_th1,gNL_th1,logAs1,tau1 = output_bessel(get_b(nr[2*l+1]),dr[2*i+1],cr[2*m+1],lnvr[2*k+1],bessel_r,bessel_Bool)
                        fNL_th2,gNL_th2,logAs2,tau2 = output_bessel(get_b(nr[2*l]),dr[2*i+1],cr[2*m+1],lnvr[2*k+1],bessel_r,bessel_Bool)
                        fNL_th3,gNL_th3,logAs3,tau3 = output_bessel(get_b(nr[2*l+1]),dr[2*i],cr[2*m+1],lnvr[2*k+1],bessel_r,bessel_Bool)
                        fNL_th4,gNL_th4,logAs4,tau4 = output_bessel(get_b(nr[2*l+1]),dr[2*i+1],cr[2*m],lnvr[2*k+1],bessel_r,bessel_Bool)
                        fNL_th5,gNL_th5,logAs5,tau5 = output_bessel(get_b(nr[2*l+1]),dr[2*i+1],cr[2*m+1],lnvr[2*k],bessel_r,bessel_Bool)
                        fNL_th6,gNL_th6,logAs6,tau6 = output_bessel(get_b(nr[2*l]),dr[2*i],cr[2*m+1],lnvr[2*k+1],bessel_r,bessel_Bool)
                        fNL_th7,gNL_th7,logAs7,tau7 = output_bessel(get_b(nr[2*l]),dr[2*i+1],cr[2*m],lnvr[2*k+1],bessel_r,bessel_Bool)
                        fNL_th8,gNL_th8,logAs8,tau8 = output_bessel(get_b(nr[2*l]),dr[2*i+1],cr[2*m+1],lnvr[2*k],bessel_r,bessel_Bool)
                        
                        fNL_th9,gNL_th9,logAs9,tau9 = output_bessel(get_b(nr[2*l+1]),dr[2*i],cr[2*m],lnvr[2*k+1],bessel_r,bessel_Bool)
                        fNL_th10,gNL_th10,logAs10,tau10 = output_bessel(get_b(nr[2*l+1]),dr[2*i],cr[2*m+1],lnvr[2*k],bessel_r,bessel_Bool)
                        fNL_th11,gNL_th11,logAs11,tau11 = output_bessel(get_b(nr[2*l+1]),dr[2*i+1],cr[2*m],lnvr[2*k],bessel_r,bessel_Bool)
                        fNL_th12,gNL_th12,logAs12,tau12 = output_bessel(get_b(nr[2*l]),dr[2*i],cr[2*m],lnvr[2*k+1],bessel_r,bessel_Bool)
                        fNL_th13,gNL_th13,logAs13,tau13 = output_bessel(get_b(nr[2*l]),dr[2*i],cr[2*m+1],lnvr[2*k],bessel_r,bessel_Bool)
                        fNL_th14,gNL_th14,logAs14,tau14 = output_bessel(get_b(nr[2*l]),dr[2*i+1],cr[2*m],lnvr[2*k],bessel_r,bessel_Bool)
                        fNL_th15,gNL_th15,logAs15,tau15 = output_bessel(get_b(nr[2*l+1]),dr[2*i],cr[2*m],lnvr[2*k],bessel_r,bessel_Bool)
                        fNL_th16,gNL_th16,logAs16,tau16 = output_bessel(get_b(nr[2*l]),dr[2*i],cr[2*m],lnvr[2*k],bessel_r,bessel_Bool)
                        
    
                        alphas_th1 = ekphase(nr[2*l+1],get_b(nr[2*l+1]))[1]
                        alphas_th2 = ekphase(nr[2*l],get_b(nr[2*l]))[1]

                        

                        def like(logAs,tau,alphas_th,fNL_th,gNL_th):
                            log10_10As_th = log(1e10*exp(logAs+ log((tau*kval/2)**(ns-1))))
                            yd_minus_yth = [log10_10As - log10_10As_th,ns-ns_th,alpha-alphas_th,fNL_bestfit-fNL_th,gNL_bestfit-gNL_th]
                            this = np.dot(inverse_primordial_covariance,yd_minus_yth)
                            chisq =  np.dot(np.transpose(yd_minus_yth),this)
                            return exp(-.5*chisq)
                        
                        
                        L0 = like_prev[i][l][k][m]
                        L1 = like(logAs1,tau1,alphas_th1,fNL_th1,gNL_th1)
                        L2 = like(logAs2,tau2,alphas_th2,fNL_th2,gNL_th2)
                        L3 = like(logAs3,tau3,alphas_th1,fNL_th3,gNL_th3)
                        L4 = like(logAs4,tau4,alphas_th1,fNL_th4,gNL_th4)
                        L5 = like(logAs5,tau5,alphas_th1,fNL_th5,gNL_th5)
                        L6 = like(logAs6,tau6,alphas_th2,fNL_th6,gNL_th6)
                        L7 = like(logAs7,tau7,alphas_th2,fNL_th7,gNL_th7)
                        L8 = like(logAs8,tau8,alphas_th2,fNL_th8,gNL_th8)
                        L9 = like(logAs9,tau9,alphas_th1,fNL_th9,gNL_th9)
                        L10 = like(logAs10,tau10,alphas_th1,fNL_th10,gNL_th10)
                        L11 = like(logAs11,tau11,alphas_th1,fNL_th11,gNL_th11)
                        L12 = like(logAs12,tau12,alphas_th2,fNL_th12,gNL_th12)
                        L13 = like(logAs13,tau13,alphas_th2,fNL_th13,gNL_th13)
                        L14 = like(logAs14,tau14,alphas_th2,fNL_th14,gNL_th14)
                        L15 = like(logAs15,tau15,alphas_th1,fNL_th15,gNL_th15)
                        L16 = like(logAs16,tau16,alphas_th2,fNL_th16,gNL_th16)
                        
                        likelihoodr[2*i+1][2*l+1][2*k+1][2*m+1] = L1
                        likelihoodr[2*i+1][2*l][2*k+1][2*m+1] = L2
                        likelihoodr[2*i][2*l+1][2*k+1][2*m+1] = L3
                        likelihoodr[2*i+1][2*l+1][2*k+1][2*m] = L4
                        likelihoodr[2*i+1][2*l+1][2*k][2*m+1] = L5
                        likelihoodr[2*i][2*l][2*k+1][2*m+1] = L6
                        likelihoodr[2*i+1][2*l][2*k+1][2*m] = L7
                        likelihoodr[2*i+1][2*l][2*k][2*m+1] = L8
                        likelihoodr[2*i][2*l+1][2*k+1][2*m] = L9
                        likelihoodr[2*i][2*l+1][2*k][2*m+1] = L10
                        likelihoodr[2*i+1][2*l+1][2*k][2*m] = L11
                        likelihoodr[2*i][2*l][2*k+1][2*m] = L12
                        likelihoodr[2*i][2*l][2*k][2*m+1] = L13
                        likelihoodr[2*i+1][2*l][2*k][2*m] = L14
                        likelihoodr[2*i][2*l+1][2*k][2*m] = L15
                        likelihoodr[2*i][2*l][2*k][2*m] = L16
                        
                        fNLr[2*i+1][2*l+1][2*k+1][2*m+1] = fNL_th1
                        fNLr[2*i+1][2*l][2*k+1][2*m+1] = fNL_th2
                        fNLr[2*i][2*l+1][2*k+1][2*m+1] = fNL_th3
                        fNLr[2*i+1][2*l+1][2*k+1][2*m] = fNL_th4
                        fNLr[2*i+1][2*l+1][2*k][2*m+1] = fNL_th5
                        fNLr[2*i][2*l][2*k+1][2*m+1] = fNL_th6
                        fNLr[2*i+1][2*l][2*k+1][2*m] = fNL_th7
                        fNLr[2*i+1][2*l][2*k][2*m+1] = fNL_th8
                        fNLr[2*i][2*l+1][2*k+1][2*m] = fNL_th9
                        fNLr[2*i][2*l+1][2*k][2*m+1] = fNL_th10
                        fNLr[2*i+1][2*l+1][2*k][2*m] = fNL_th11
                        fNLr[2*i][2*l][2*k+1][2*m] = fNL_th12
                        fNLr[2*i][2*l][2*k][2*m+1] = fNL_th13
                        fNLr[2*i+1][2*l][2*k][2*m] = fNL_th14
                        fNLr[2*i][2*l+1][2*k][2*m] = fNL_th15
                        fNLr[2*i][2*l][2*k][2*m] = fNL_th16
                        
                        gNLr[2*i+1][2*l+1][2*k+1][2*m+1] = gNL_th1
                        gNLr[2*i+1][2*l][2*k+1][2*m+1] = gNL_th2
                        gNLr[2*i][2*l+1][2*k+1][2*m+1] = gNL_th3
                        gNLr[2*i+1][2*l+1][2*k+1][2*m] = gNL_th4
                        gNLr[2*i+1][2*l+1][2*k][2*m+1] = gNL_th5
                        gNLr[2*i][2*l][2*k+1][2*m+1] = gNL_th6
                        gNLr[2*i+1][2*l][2*k+1][2*m] = gNL_th7
                        gNLr[2*i+1][2*l][2*k][2*m+1] = gNL_th8
                        gNLr[2*i][2*l+1][2*k+1][2*m] = gNL_th9
                        gNLr[2*i][2*l+1][2*k][2*m+1] = gNL_th10
                        gNLr[2*i+1][2*l+1][2*k][2*m] = gNL_th11
                        gNLr[2*i][2*l][2*k+1][2*m] = gNL_th12
                        gNLr[2*i][2*l][2*k][2*m+1] = gNL_th13
                        gNLr[2*i+1][2*l][2*k][2*m] = gNL_th14
                        gNLr[2*i][2*l+1][2*k][2*m] = gNL_th15
                        gNLr[2*i][2*l][2*k][2*m] = gNL_th16
                        
                        #Now have everything for this cell, so spline and integrate
                        vals = np.array([L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15,L16])
                        x1 = np.array([dhere[i] , dr[2*i+1], dr[2*i+1],dr[2*i],dr[2*i+1],dr[2*i+1],dr[2*i],dr[2*i+1],dr[2*i+1],dr[2*i],dr[2*i],dr[2*i+1],dr[2*i],dr[2*i],dr[2*i+1],dr[2*i],dr[2*i] ])
                        x2 = np.array([nhere[l] , nr[2*l+1],nr[2*l],nr[2*l+1],nr[2*l+1],nr[2*l+1],nr[2*l],nr[2*l],nr[2*l],nr[2*l+1],nr[2*l+1],nr[2*l+1],nr[2*l],nr[2*l],nr[2*l],nr[2*l+1],nr[2*l]])
                        x3 = np.array([lnvhere[k] , lnvr[2*k+1],lnvr[2*k+1],lnvr[2*k+1],lnvr[2*k+1],lnvr[2*k],lnvr[2*k+1],lnvr[2*k+1],lnvr[2*k],lnvr[2*k+1],lnvr[2*k],lnvr[2*k],lnvr[2*k+1],lnvr[2*k],lnvr[2*k],lnvr[2*k],lnvr[2*k]])
                        x4 = np.array([chere[m], cr[2*m+1],cr[2*m+1],cr[2*m+1],cr[2*m],cr[2*m+1],cr[2*m+1],cr[2*m],cr[2*m+1],cr[2*m],cr[2*m+1],cr[2*m],cr[2*m],cr[2*m+1],cr[2*m],cr[2*m],cr[2*m]])
                        pts = list(zip(x1,x2,x3,x4))
                      
                      
                        ####NEW interp
                        interp = LinearNDInterpolator(pts,vals)
                        logvals = np.zeros([len(vals)])
                        for ii in range(17):
                            if vals[ii] > 0:
                                logvals[ii] = vals[ii]*np.log(vals[ii])
                            else:
                                logvals[ii] = 0 
                        interp_log = LinearNDInterpolator(pts,logvals)
                        NN =10
                        x1 = np.linspace(dr[2*i],dr[2*i+1],NN)
                        x2 = np.linspace(nr[2*l],nr[2*l+1],NN)
                        x3 = np.linspace(lnvr[2*k],lnvr[2*k+1],NN)
                        x4 = np.linspace(cr[2*m],cr[2*m+1],NN)
                        
                        
                        factor = [10/(x1[-1] - x1[0]) ,10/(x2[-1] - x2[0]),10/(x3[-1] - x3[0]),10/(x4[-1] - x4[0]) ]
                        v = interp(x1[:,None,None,None],x2[None,:,None,None],x3[None,None,:,None],x4[None,None,None,:])
                        v_log = interp_log(x1[:,None,None,None],x2[None,:,None,None],x3[None,None,:,None],x4[None,None,None,:])
                        a = np.mgrid[x1[0]*factor[0]:x1[-1]*factor[0], x2[0]*factor[1]:x2[-1]*factor[1],x3[0]*factor[2]:x3[-1]*factor[2],x4[0]*factor[3]:x4[-1]*factor[3]] 
                        a = np.rollaxis(a, 0, 5)         # Make 0th axis into the last axis
                        a[:,:,:,:,0] = a[:,:,:,:,0] / factor[0]
                        a[:,:,:,:,1] = a[:,:,:,:,1] / factor[1]
                        a[:,:,:,:,2] = a[:,:,:,:,2] / factor[2]
                        a[:,:,:,:,3] = a[:,:,:,:,3] / factor[3]
                        M = np.shape(a)[1]
                        O = np.shape(a)[0]
                        P = np.shape(a)[2]
                        Q = np.shape(a)[3]
                        a = a.reshape((M * O * P *Q, 4))   # Now reshape while preserving order

                        for p in range(M * O * P *Q):
                            if a[p,0] < x1[0]:
                                a[p,0] = x1[0]
                            if a[p,0] > x1[-1]:
                                a[p,0] = x1[-1]
                            if a[p,1] < x2[0]:
                                a[p,1] = x2[0]
                            if a[p,1] > x2[-1]:
                                a[p,1] = x2[-1]
                            if a[p,2] < x3[0]:
                                a[p,2] = x3[0]
                            if a[p,2] > x3[-1]:
                                a[p,2] = x3[-1]
                            if a[p,3] < x4[0]:
                                a[p,3] = x4[0]
                            if a[p,3] > x4[-1]:
                                a[p,3] = x4[-1]
                        
                        interp_v = interpn((x1, x2, x3, x4), v, a)
                        interp_log = interpn((x1, x2, x3, x4), v_log, a)
                        x1small = np.linspace(x1[0], x1[-1], O)
                        x2small = np.linspace(x2[0], x2[-1], M)
                        x3small = np.linspace(x3[0], x3[-1], P)
                        x4small = np.linspace(x4[0], x4[-1], Q)
                        test2 = interp_v.reshape(x1small.size, x2small.size,x3small.size,x4small.size)
                        log_thing = interp_log.reshape(x1small.size, x2small.size,x3small.size,x4small.size)
                        res = simps(simps(simps(simps(test2, x4small), x3small),x2small),x1small)
                        res_log = simps(simps(simps(simps(log_thing, x4small), x3small),x2small),x1small)
                        
                        ####END interp
                        
                        Eprime += res/((c[-1]-c[0])*(d[-1]-d[0])*(lnv[-1]-lnv[0])*(n[-1]-n[0])) #s_over_n#exp(-.5*chisq)
                        Log_new += res_log/((c[-1]-c[0])*(d[-1]-d[0])*(lnv[-1]-lnv[0])*(n[-1]-n[0]))
                        numpoints += 1
                        
                        if any(y > .001 for y in vals[1:]):
                            if k == 0 and kfirst==False:
                                print('Problem? k = '+str(k))
                                kfirst = True
                            if m == 0 and mfirst==False:
                                print('Problem? m = '+str(m))
                                mfirst = True
                            if i == 0 and ifirst==False:
                                print('Problem? i = '+str(i))
                                ifirst = True
                            if i == int(2**q*start-1) and ilast==False:
                                print('Problem? i = '+str(i))
                                ilast = True
                            if l == int(2**q*start-1) and llast==False:
                                print('Problem? l = '+str(l))
                                llast = True
                            if k == int(2**q*start-1) and klast==False:
                                print('Problem? k = '+str(k))
                                klast = True
                            if m == int(2**q*start-1) and mlast==False:
                                print('Problem? m = '+str(m))
                                mlast = True

                        
                        
                        if np.any(vals > Likemax): 
                            Likemax = np.max(vals)
                            Maxloc = [dr[i],nr[l],lnvr[k],cr[m]]
                        

                    else:

                        likelihoodr[2*i+1][2*l+1][2*k+1][2*m+1] = like_prev[i][l][k][m]
                        likelihoodr[2*i+1][2*l+1][2*k+1][2*m] = like_prev[i][l][k][m]
                        likelihoodr[2*i+1][2*l+1][2*k][2*m+1] = like_prev[i][l][k][m]
                        likelihoodr[2*i+1][2*l][2*k+1][2*m+1] = like_prev[i][l][k][m]
                        likelihoodr[2*i][2*l+1][2*k+1][2*m+1] = like_prev[i][l][k][m]
                        likelihoodr[2*i+1][2*l+1][2*k][2*m] = like_prev[i][l][k][m]
                        likelihoodr[2*i+1][2*l][2*k+1][2*m] = like_prev[i][l][k][m]
                        likelihoodr[2*i][2*l+1][2*k+1][2*m] = like_prev[i][l][k][m]
                        likelihoodr[2*i+1][2*l][2*k][2*m+1] = like_prev[i][l][k][m]
                        likelihoodr[2*i][2*l+1][2*k][2*m+1] = like_prev[i][l][k][m]
                        likelihoodr[2*i][2*l][2*k+1][2*m+1] = like_prev[i][l][k][m]
                        likelihoodr[2*i][2*l][2*k][2*m+1] = like_prev[i][l][k][m]
                        likelihoodr[2*i][2*l][2*k+1][2*m] = like_prev[i][l][k][m]
                        likelihoodr[2*i][2*l+1][2*k][2*m] = like_prev[i][l][k][m]
                        likelihoodr[2*i+1][2*l][2*k][2*m] = like_prev[i][l][k][m]
                        likelihoodr[2*i][2*l][2*k][2*m] = like_prev[i][l][k][m]
                        summation_else += like_prev[i][l][k][m]
                #        if likelihoodr[i][l][k][m] > 0:
                #            L_lnL += likelihoodr[i][l][k][m]*log(likelihoodr[i][l][k][m])
                #        else:
                #            L_lnL += 0
                        numpoints +=1

    dv = lnvr[1]-lnvr[0];dc = cr[1]-cr[0];dn = nr[1]-nr[0];dd = dr[1]-dr[0]
    print('Eprime is '+str(Eprime))
    print('numpoints is '+str(numpoints))
    print('summation_else = '+str(summation_else))
    print('L_lnL / N = '+str(Log_new))
   # print('After this, Eprime = '+str(summation*dv*dc*dn*dd/((c[-1]-c[0])*(d[-1]-d[0])*(lnv[-1]-lnv[0])*(n[-1]-n[0]))))#+', and numpoints = '+str(numpoints)+', and L_lnL = '+str(L_lnL))
   # print('and sum_else divided by s_o_n = '+str(summation_else/summation))
    print('delta theta cell = '+str(dv*dc*dn*dd))
    print('delta theta code = '+str((c[-1]-c[0])*(d[-1]-d[0])*(lnv[-1]-lnv[0])*(n[-1]-n[0])))
    print('Max L is '+str(Likemax)+', at '+str(Maxloc))


    like_prev = likelihoodr
    print('q = '+str(q)+' took '+str(time.time() - begin))

1/0
q=4
likelihoodr = 0


begin = time.time()
lnvr = [lnv[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))] 
nr = [n[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))]
dr = [d[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))]
cr = [c[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))]

#likelihoodr = np.zeros([len(dr),len(nr),len(lnvr),len(cr)])
Eprime = 0
Log_new = 0
    
ilast = False
llast = False
klast = False
mlast = False
ifirst = False
mfirst = False
kfirst = False
   
    
nhere = [n[int(size/start/(2**q)*(1 + 2*i))] for i in range(int(2**(3+q)))]
chere = [c[int(size/start/(2**q)*(1 + 2*i))] for i in range(int(2**(3+q)))]
lnvhere = [lnv[int(size/start/(2**q)*(1 + 2*i))] for i in range(int(2**(3+q)))] 
dhere =[d[int(size/start/(2**q)*(1 + 2*i))] for i in range(int(2**(3+q)))]
    

for i in range(int(2**q*start)):
    if i%10 == 0:
        print('At i = '+str(i)+' , max like_ref is '+str(Likemax))

    for l in range(int(2**q*start)):

        ns_th = ns
        bhere = get_b(nhere[l])
            

        ns_th = ekphase(nhere[l],bhere)[0]
        alphas_th = ekphase(nhere[l],bhere)[1]
        for k in range(int(2**q*start)):
            for m in range(int(2**q*start)):


                if like_prev[i][l][k][m] >= threshold:
                    fNL_th1,gNL_th1,logAs1,tau1 = output_bessel(get_b(nr[2*l+1]),dr[2*i+1],cr[2*m+1],lnvr[2*k+1],bessel_r,bessel_Bool)
                    fNL_th2,gNL_th2,logAs2,tau2 = output_bessel(get_b(nr[2*l]),dr[2*i+1],cr[2*m+1],lnvr[2*k+1],bessel_r,bessel_Bool)
                    fNL_th3,gNL_th3,logAs3,tau3 = output_bessel(get_b(nr[2*l+1]),dr[2*i],cr[2*m+1],lnvr[2*k+1],bessel_r,bessel_Bool)
                    fNL_th4,gNL_th4,logAs4,tau4 = output_bessel(get_b(nr[2*l+1]),dr[2*i+1],cr[2*m],lnvr[2*k+1],bessel_r,bessel_Bool)
                    fNL_th5,gNL_th5,logAs5,tau5 = output_bessel(get_b(nr[2*l+1]),dr[2*i+1],cr[2*m+1],lnvr[2*k],bessel_r,bessel_Bool)
                    fNL_th6,gNL_th6,logAs6,tau6 = output_bessel(get_b(nr[2*l]),dr[2*i],cr[2*m+1],lnvr[2*k+1],bessel_r,bessel_Bool)
                    fNL_th7,gNL_th7,logAs7,tau7 = output_bessel(get_b(nr[2*l]),dr[2*i+1],cr[2*m],lnvr[2*k+1],bessel_r,bessel_Bool)
                    fNL_th8,gNL_th8,logAs8,tau8 = output_bessel(get_b(nr[2*l]),dr[2*i+1],cr[2*m+1],lnvr[2*k],bessel_r,bessel_Bool)
                        
                    fNL_th9,gNL_th9,logAs9,tau9 = output_bessel(get_b(nr[2*l+1]),dr[2*i],cr[2*m],lnvr[2*k+1],bessel_r,bessel_Bool)
                    fNL_th10,gNL_th10,logAs10,tau10 = output_bessel(get_b(nr[2*l+1]),dr[2*i],cr[2*m+1],lnvr[2*k],bessel_r,bessel_Bool)
                    fNL_th11,gNL_th11,logAs11,tau11 = output_bessel(get_b(nr[2*l+1]),dr[2*i+1],cr[2*m],lnvr[2*k],bessel_r,bessel_Bool)
                    fNL_th12,gNL_th12,logAs12,tau12 = output_bessel(get_b(nr[2*l]),dr[2*i],cr[2*m],lnvr[2*k+1],bessel_r,bessel_Bool)
                    fNL_th13,gNL_th13,logAs13,tau13 = output_bessel(get_b(nr[2*l]),dr[2*i],cr[2*m+1],lnvr[2*k],bessel_r,bessel_Bool)
                    fNL_th14,gNL_th14,logAs14,tau14 = output_bessel(get_b(nr[2*l]),dr[2*i+1],cr[2*m],lnvr[2*k],bessel_r,bessel_Bool)
                    fNL_th15,gNL_th15,logAs15,tau15 = output_bessel(get_b(nr[2*l+1]),dr[2*i],cr[2*m],lnvr[2*k],bessel_r,bessel_Bool)
                    fNL_th16,gNL_th16,logAs16,tau16 = output_bessel(get_b(nr[2*l]),dr[2*i],cr[2*m],lnvr[2*k],bessel_r,bessel_Bool)
                        
    
                    alphas_th1 = ekphase(nr[2*l+1],get_b(nr[2*l+1]))[1]
                    alphas_th2 = ekphase(nr[2*l],get_b(nr[2*l]))[1]

                        

                    def like(logAs,tau,alphas_th,fNL_th,gNL_th):
                        log10_10As_th = log(1e10*exp(logAs+ log((tau*kval/2)**(ns-1))))
                        yd_minus_yth = [log10_10As - log10_10As_th,ns-ns_th,alpha-alphas_th,fNL_bestfit-fNL_th,gNL_bestfit-gNL_th]
                        this = np.dot(inverse_primordial_covariance,yd_minus_yth)
                        chisq =  np.dot(np.transpose(yd_minus_yth),this)
                        return exp(-.5*chisq)
                        
                        
                    L0 = like_prev[i][l][k][m]
                    L1 = like(logAs1,tau1,alphas_th1,fNL_th1,gNL_th1)
                    L2 = like(logAs2,tau2,alphas_th2,fNL_th2,gNL_th2)
                    L3 = like(logAs3,tau3,alphas_th1,fNL_th3,gNL_th3)
                    L4 = like(logAs4,tau4,alphas_th1,fNL_th4,gNL_th4)
                    L5 = like(logAs5,tau5,alphas_th1,fNL_th5,gNL_th5)
                    L6 = like(logAs6,tau6,alphas_th2,fNL_th6,gNL_th6)
                    L7 = like(logAs7,tau7,alphas_th2,fNL_th7,gNL_th7)
                    L8 = like(logAs8,tau8,alphas_th2,fNL_th8,gNL_th8)
                    L9 = like(logAs9,tau9,alphas_th1,fNL_th9,gNL_th9)
                    L10 = like(logAs10,tau10,alphas_th1,fNL_th10,gNL_th10)
                    L11 = like(logAs11,tau11,alphas_th1,fNL_th11,gNL_th11)
                    L12 = like(logAs12,tau12,alphas_th2,fNL_th12,gNL_th12)
                    L13 = like(logAs13,tau13,alphas_th2,fNL_th13,gNL_th13)
                    L14 = like(logAs14,tau14,alphas_th2,fNL_th14,gNL_th14)
                    L15 = like(logAs15,tau15,alphas_th1,fNL_th15,gNL_th15)
                    L16 = like(logAs16,tau16,alphas_th2,fNL_th16,gNL_th16)
                        
                    
                        #Now have everything for this cell, so spline and integrate
                    vals = np.array([L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15,L16])
                    x1 = np.array([dhere[i] , dr[2*i+1], dr[2*i+1],dr[2*i],dr[2*i+1],dr[2*i+1],dr[2*i],dr[2*i+1],dr[2*i+1],dr[2*i],dr[2*i],dr[2*i+1],dr[2*i],dr[2*i],dr[2*i+1],dr[2*i],dr[2*i] ])
                    x2 = np.array([nhere[l] , nr[2*l+1],nr[2*l],nr[2*l+1],nr[2*l+1],nr[2*l+1],nr[2*l],nr[2*l],nr[2*l],nr[2*l+1],nr[2*l+1],nr[2*l+1],nr[2*l],nr[2*l],nr[2*l],nr[2*l+1],nr[2*l]])
                    x3 = np.array([lnvhere[k] , lnvr[2*k+1],lnvr[2*k+1],lnvr[2*k+1],lnvr[2*k+1],lnvr[2*k],lnvr[2*k+1],lnvr[2*k+1],lnvr[2*k],lnvr[2*k+1],lnvr[2*k],lnvr[2*k],lnvr[2*k+1],lnvr[2*k],lnvr[2*k],lnvr[2*k],lnvr[2*k]])
                    x4 = np.array([chere[m], cr[2*m+1],cr[2*m+1],cr[2*m+1],cr[2*m],cr[2*m+1],cr[2*m+1],cr[2*m],cr[2*m+1],cr[2*m],cr[2*m+1],cr[2*m],cr[2*m],cr[2*m+1],cr[2*m],cr[2*m],cr[2*m]])
                    pts = list(zip(x1,x2,x3,x4))
                      
                      
                        ####NEW interp
                    interp = LinearNDInterpolator(pts,vals)
                    logvals = np.zeros([len(vals)])
                    for ii in range(17):
                        if vals[ii] > 0:
                            logvals[ii] = vals[ii]*np.log(vals[ii])
                        else:
                            logvals[ii] = 0
                    interp_log = LinearNDInterpolator(pts,logvals)
                    NN =10
                    x1 = np.linspace(dr[2*i],dr[2*i+1],NN)
                    x2 = np.linspace(nr[2*l],nr[2*l+1],NN)
                    x3 = np.linspace(lnvr[2*k],lnvr[2*k+1],NN)
                    x4 = np.linspace(cr[2*m],cr[2*m+1],NN)
                        
                        
                    factor = [10/(x1[-1] - x1[0]) ,10/(x2[-1] - x2[0]),10/(x3[-1] - x3[0]),10/(x4[-1] - x4[0]) ]
                    v = interp(x1[:,None,None,None],x2[None,:,None,None],x3[None,None,:,None],x4[None,None,None,:])
                    v_log = interp_log(x1[:,None,None,None],x2[None,:,None,None],x3[None,None,:,None],x4[None,None,None,:])
                    a = np.mgrid[x1[0]*factor[0]:x1[-1]*factor[0], x2[0]*factor[1]:x2[-1]*factor[1],x3[0]*factor[2]:x3[-1]*factor[2],x4[0]*factor[3]:x4[-1]*factor[3]] 
                    a = np.rollaxis(a, 0, 5)         # Make 0th axis into the last axis
                    a[:,:,:,:,0] = a[:,:,:,:,0] / factor[0]
                    a[:,:,:,:,1] = a[:,:,:,:,1] / factor[1]
                    a[:,:,:,:,2] = a[:,:,:,:,2] / factor[2]
                    a[:,:,:,:,3] = a[:,:,:,:,3] / factor[3]
                    M = np.shape(a)[1]
                    O = np.shape(a)[0]
                    P = np.shape(a)[2]
                    Q = np.shape(a)[3]
                    a = a.reshape((M * O * P *Q, 4))   # Now reshape while preserving order

                    for p in range(M * O * P *Q):
                        if a[p,0] < x1[0]:
                            a[p,0] = x1[0]
                        if a[p,0] > x1[-1]:
                            a[p,0] = x1[-1]
                        if a[p,1] < x2[0]:
                            a[p,1] = x2[0]
                        if a[p,1] > x2[-1]:
                            a[p,1] = x2[-1]
                        if a[p,2] < x3[0]:
                            a[p,2] = x3[0]
                        if a[p,2] > x3[-1]:
                            a[p,2] = x3[-1]
                        if a[p,3] < x4[0]:
                            a[p,3] = x4[0]
                        if a[p,3] > x4[-1]:
                            a[p,3] = x4[-1]
                        
                    interp_v = interpn((x1, x2, x3, x4), v, a)
                    interp_log = interpn((x1, x2, x3, x4), v_log, a)
                    x1small = np.linspace(x1[0], x1[-1], O)
                    x2small = np.linspace(x2[0], x2[-1], M)
                    x3small = np.linspace(x3[0], x3[-1], P)
                    x4small = np.linspace(x4[0], x4[-1], Q)
                    test2 = interp_v.reshape(x1small.size, x2small.size,x3small.size,x4small.size)
                    log_thing = interp_log.reshape(x1small.size, x2small.size,x3small.size,x4small.size)
                    res = simps(simps(simps(simps(test2, x4small), x3small),x2small),x1small)
                    res_log = simps(simps(simps(simps(log_thing, x4small), x3small),x2small),x1small)
                        
                        ####END interp
                        
                    Eprime += res/((c[-1]-c[0])*(d[-1]-d[0])*(lnv[-1]-lnv[0])*(n[-1]-n[0])) #s_over_n#exp(-.5*chisq)
                    Log_new += res_log/((c[-1]-c[0])*(d[-1]-d[0])*(lnv[-1]-lnv[0])*(n[-1]-n[0]))
                    numpoints += 1
                        
                    if any(y > .001 for y in vals[1:]):
                        if k == 0 and kfirst==False:
                            print('Problem? k = '+str(k))
                            kfirst = True
                        if m == 0 and mfirst==False:
                            print('Problem? m = '+str(m))
                            mfirst = True
                        if i == 0 and ifirst==False:
                            print('Problem? i = '+str(i))
                            ifirst = True
                        if i == int(2**q*start-1) and ilast==False:
                            print('Problem? i = '+str(i))
                            ilast = True
                        if l == int(2**q*start-1) and llast==False:
                            print('Problem? l = '+str(l))
                            llast = True
                        if k == int(2**q*start-1) and klast==False:
                            print('Problem? k = '+str(k))
                            klast = True
                        if m == int(2**q*start-1) and mlast==False:
                            print('Problem? m = '+str(m))
                            mlast = True

                        
                     #THIS COULD BE WRONG? MIGHT NOT GIVE RIGHT COORDS
                     #I DONT ACTUALLY WANT THE CURRENT i.l.k.m , I 
                     #want the specific coords that np.max(vals) happens at
                    if np.any(vals > Likemax): 
                        Likemax = np.max(vals)
                        Maxloc = [dr[i],nr[l],lnvr[k],cr[m]]
                        

                else:

                    summation_else += like_prev[i][l][k][m]
                #        if likelihoodr[i][l][k][m] > 0:
                #            L_lnL += likelihoodr[i][l][k][m]*log(likelihoodr[i][l][k][m])
                #        else:
                #            L_lnL += 0
                    numpoints +=1

dv = lnvr[1]-lnvr[0];dc = cr[1]-cr[0];dn = nr[1]-nr[0];dd = dr[1]-dr[0]
print('Eprime is '+str(Eprime))
print('numpoints is '+str(numpoints))
print('summation_else = '+str(summation_else))
print('L_lnL / N = '+str(Log_new))
   # print('After this, Eprime = '+str(summation*dv*dc*dn*dd/((c[-1]-c[0])*(d[-1]-d[0])*(lnv[-1]-lnv[0])*(n[-1]-n[0]))))#+', and numpoints = '+str(numpoints)+', and L_lnL = '+str(L_lnL))
   # print('and sum_else divided by s_o_n = '+str(summation_else/summation))
print('delta theta cell = '+str(dv*dc*dn*dd))
print('delta theta code = '+str((c[-1]-c[0])*(d[-1]-d[0])*(lnv[-1]-lnv[0])*(n[-1]-n[0])))
print('Max L is '+str(Likemax)+', at '+str(Maxloc))


#like_prev = likelihoodr
print('q = '+str(q)+' took '+str(time.time() - begin))
