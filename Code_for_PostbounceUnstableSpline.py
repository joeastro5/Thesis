#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 02:32:57 2021

@author: joe
"""

import numpy as np
from math import *
from matplotlib import tri
from Integ_deltasUnstable import new_result
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
                                  [0,0,0,5.1**2,0],
                                  [0,0,0,0,(6.5e4)**2]])#np.array([[samples[4][4],samples[4][5]],[samples[5][4],samples[5][5]]])
#eg 12.9 https://wiki.cosmos.esa.int/planck-legacy-archive/images/4/43/Baseline_params_table_2018_68pc_v2.pdf in says 0.0046, which squared is about 2e-5. This is for 2018 data, but clsoe enough

inverse_primordial_covariance = np.linalg.inv(primordial_covariance)
ns = 0.9635
log10_10As = 3.049
alpha = -0.0055
fNL_bestfit = -.9 #2018
gNL_bestfit = -5.8e4
kval = 1e-30
Likemax = 0 #Keeping track of max Like found so far
Maxloc = [0,0,0,0]
#######################################


def ekphase(n,N = 60):
    this = 2/(ns-1+n/(N+1))/(N+1)**n #This makes ns = ns_Planck
    eps_ek = this*(N+1)**n#exp(lneps)*(N+1)**n
    epsN = n*eps_ek/(N+1)
    epsNN = n*(n-1)*eps_ek/(N+1)**2
    c = sqrt(2*eps_ek) 
    nS = 1.+2/eps_ek - epsN/eps_ek
    alphaS = 2*epsN/eps_ek**2 + (epsNN/eps_ek - (epsN/eps_ek)**2) 
    return [nS,alphaS,c]

#def get_b(n,eps,nS=ns):
#    return sqrt(2*eps*61**n)*(1+(nS-1+7/3*n/61)/(-2/(1-1/(eps*61**n))))

start = 8
size = 256
dim = 2*size

lneps = np.linspace(4,6,dim)#(log(7),log(4000),dim)
n = np.linspace(2.23,3,dim)
logV = np.linspace(-18,-8,dim)
q = np.linspace(0.9,1.25,dim)#0.,1.25,dim)

lnepshere = [lneps[int(size/start + i*dim/start)] for i in range(start)]
qhere = [q[int(size/start + i*dim/start)] for i in range(start)]
logVhere = [logV[int(size/start + i*dim/start)] for i in range(start)]
nhere = [n[int(size/start + i*dim/start)] for i in range(start)]

summation = 0
numpoints = 0
L_lnL = 0
Likemax = 0

likelihood = np.zeros([start,start,start,start])
ifirst = False
jfirst = False
kfirst = False
lfirst = False
ilast = False
jlast = False
klast = False
llast = False

for i in range(start):  #lneps
    print('i = '+str(i)+', max like is '+str(np.max(likelihood)))
    for j in range(start): #q
        #print('j = '+str(j))
        for k in range(start): #logV
            fNL,gNL,logAs,tau = new_result(exp(lnepshere[i]),qhere[j],logVhere[k])

            for l in range(start):  #n
                
                
                eps_ek = exp(lnepshere[i])*(61)**nhere[l]
                c_ek = sqrt(2*eps_ek)
                alpha_th = ekphase(nhere[l])[1] 
                ns_th = ekphase(nhere[l])[0] #ns
                log10_10As_th = log(1e10*exp(logAs+ log((tau*kval/2)**(ns-1))))

                    
                yd_minus_yth = [log10_10As - log10_10As_th,ns-ns_th,alpha - alpha_th,fNL_bestfit-fNL,gNL_bestfit-gNL]

                this = np.dot(inverse_primordial_covariance,yd_minus_yth)
                chisq =  np.dot(np.transpose(yd_minus_yth),this)  # is it < dof?        
                likelihood[i][j][k][l]= exp(-.5*chisq)
                summation +=exp(-.5*chisq)
                numpoints += 1
                L_lnL += exp(-.5*chisq)*(-.5*chisq)
                if exp(-.5*chisq) > .01:
                    if i == 0 and ifirst==False:
                        print('Problem? i = '+str(i))
                        ifirst = True
                   # if j == 0 and jfirst==False:
                   #     print('Problem? j = '+str(j))
                   #     jfirst = True
                    if k == 0 and kfirst==False:
                        print('Problem? k = '+str(k))
                        kfirst = True
                   # if l == 0 and lfirst==False:
                   #     print('Problem? l = '+str(l))
                   #     lfirst = True
                    if i == start-1 and ilast==False:
                        print('Problem? i = '+str(i))
                        ilast = True
                   # if j == start-1 and jlast==False:
                   #     print('Problem? j = '+str(j))
                   #     jlast = True
                    if k == start-1 and klast==False:
                        print('Problem? k = '+str(k))
                        klast = True
                    if l == start-1 and llast==False:
                        print('Problem? l = '+str(l))
                        llast = True

                if exp(-.5*chisq) > Likemax:
                    Likemax = exp(-.5*chisq)
                    Maxloc = [lnepshere[i],qhere[j],logVhere[k],nhere[l]]
                    epsmaxindex = i;qmaxindex = j;Vmaxindex = k;nmaxindex = l



like_prev = likelihood
threshold = 1e-8
summation_else = 0
Eprime = summation/numpoints 
Log_new = L_lnL/numpoints
print('After, s over n (which is Eprime) is '+str(Eprime))
print('And L_lnL /N = '+str(Log_new))
for z in range(4): 
    begin = time.time()
    lnepsr = [lneps[int(size/start/(2**(z+1))*(1 + 2*i))] for i in range(int(2**(4+z)))] 
    nr = [n[int(size/start/(2**(z+1))*(1 + 2*i))] for i in range(int(2**(4+z)))]
    logVr = [logV[int(size/start/(2**(z+1))*(1 + 2*i))] for i in range(int(2**(4+z)))]
    qr = [q[int(size/start/(2**(z+1))*(1 + 2*i))] for i in range(int(2**(4+z)))]

    likelihoodr = np.zeros([len(lnepsr),len(qr),len(logVr),len(nr)])
    Eprime = 0
    Log_new = 0
    
    ifirst = False
    lfirst = False
    kfirst = False
    mfirst = False
    ilast = False
    llast = False
    klast = False
    mlast = False
    
    lnepshere = [lneps[int(size/start/(2**z)*(1 + 2*i))] for i in range(int(2**(3+z)))]
    nhere = [n[int(size/start/(2**z)*(1 + 2*i))] for i in range(int(2**(3+z)))]
    logVhere = [logV[int(size/start/(2**z)*(1 + 2*i))] for i in range(int(2**(3+z)))]
    qhere =[q[int(size/start/(2**z)*(1 + 2*i))] for i in range(int(2**(3+z)))]

    
    for i in range(int(2**z*start)):#lneps
        if i%10==0:
            print('At i = '+str(i)+' , max like_ref is '+str(np.max(likelihoodr)))

        for l in range(int(2**z*start)):#q

            
            for k in range(int(2**z*start)):#lnV
                if any(y>threshold for y in like_prev[i][l][k]):
                    
                    fNL_th1,gNL_th1,logAs1,tau1 = new_result(exp(lnepsr[2*i+1]),qr[2*l+1],logVr[2*k+1])
                    fNL_th2,gNL_th2,logAs2,tau2 = new_result(exp(lnepsr[2*i]),qr[2*l+1],logVr[2*k+1])
                    fNL_th3,gNL_th3,logAs3,tau3 = new_result(exp(lnepsr[2*i+1]),qr[2*l],logVr[2*k+1])
                    fNL_th4,gNL_th4,logAs4,tau4 = new_result(exp(lnepsr[2*i+1]),qr[2*l+1],logVr[2*k])
                    fNL_th5,gNL_th5,logAs5,tau5 = new_result(exp(lnepsr[2*i]),qr[2*l],logVr[2*k+1])
                    fNL_th6,gNL_th6,logAs6,tau6 = new_result(exp(lnepsr[2*i]),qr[2*l+1],logVr[2*k])
                    fNL_th7,gNL_th7,logAs7,tau7 = new_result(exp(lnepsr[2*i+1]),qr[2*l],logVr[2*k])
                    fNL_th8,gNL_th8,logAs8,tau8 = new_result(exp(lnepsr[2*i]),qr[2*l],logVr[2*k])
                   
                    for m in range(int(2**z*start)):#n


                    
                        ns_th = ekphase(nr[m])[0]#ns
                        #bhere = get_b(nr[m],exp(lnepsr[i]))
                        eps_ek= exp(lnepsr[i])*(61)**nr[m]
                        c_ek = sqrt(2*eps_ek)
                        
                        alphas_th = ekphase(nr[m])[1]
                        alphas_th1 = ekphase(nr[2*m+1])[1]
                        alphas_th2 = ekphase(nr[2*m])[1]
                        
                        def like(logAs,tau,alphas_th,fNL_th,gNL_th):
                            log10_10As_th = log(1e10*exp(logAs+ log((tau*kval/2)**(ns-1))))
                            yd_minus_yth = [log10_10As - log10_10As_th,ns-ns_th,alpha-alphas_th,fNL_bestfit-fNL_th,gNL_bestfit-gNL_th]
                            this = np.dot(inverse_primordial_covariance,yd_minus_yth)
                            chisq =  np.dot(np.transpose(yd_minus_yth),this)
                            return exp(-.5*chisq)


                        L0 = like_prev[i][l][k][m]
                        L1 = like(logAs1,tau1,alphas_th1,fNL_th1,gNL_th1)
                        L2 = like(logAs2,tau2,alphas_th1,fNL_th2,gNL_th2)
                        L3 = like(logAs3,tau3,alphas_th1,fNL_th3,gNL_th3)
                        L4 = like(logAs4,tau4,alphas_th1,fNL_th4,gNL_th4)
                        L5 = like(logAs5,tau5,alphas_th1,fNL_th5,gNL_th5)
                        L6 = like(logAs6,tau6,alphas_th1,fNL_th6,gNL_th6)
                        L7 = like(logAs7,tau7,alphas_th1,fNL_th7,gNL_th7)
                        L8 = like(logAs8,tau8,alphas_th1,fNL_th8,gNL_th8)
                        L9 = like(logAs1,tau1,alphas_th2,fNL_th1,gNL_th1)
                        L10 = like(logAs2,tau2,alphas_th2,fNL_th2,gNL_th2)
                        L11 = like(logAs3,tau3,alphas_th2,fNL_th3,gNL_th3)
                        L12 = like(logAs4,tau4,alphas_th2,fNL_th4,gNL_th4)
                        L13 = like(logAs5,tau5,alphas_th2,fNL_th5,gNL_th5)
                        L14 = like(logAs6,tau6,alphas_th2,fNL_th6,gNL_th6)
                        L15 = like(logAs7,tau7,alphas_th2,fNL_th7,gNL_th7)
                        L16 = like(logAs8,tau8,alphas_th2,fNL_th8,gNL_th8)
                        
                        likelihoodr[2*i+1][2*l+1][2*k+1][2*m+1] = L1
                        likelihoodr[2*i][2*l+1][2*k+1][2*m+1] = L2
                        likelihoodr[2*i+1][2*l][2*k+1][2*m+1] = L3
                        likelihoodr[2*i+1][2*l+1][2*k][2*m+1] = L4
                        likelihoodr[2*i][2*l][2*k+1][2*m+1] = L5
                        likelihoodr[2*i][2*l+1][2*k][2*m+1] = L6
                        likelihoodr[2*i+1][2*l][2*k][2*m+1] = L7
                        likelihoodr[2*i][2*l][2*k][2*m+1] = L8
                        likelihoodr[2*i+1][2*l+1][2*k+1][2*m] = L9
                        likelihoodr[2*i][2*l+1][2*k+1][2*m] = L10
                        likelihoodr[2*i+1][2*l][2*k+1][2*m] = L11
                        likelihoodr[2*i+1][2*l+1][2*k][2*m] = L12
                        likelihoodr[2*i][2*l][2*k+1][2*m] = L13
                        likelihoodr[2*i][2*l+1][2*k][2*m] = L14
                        likelihoodr[2*i+1][2*l][2*k][2*m] = L15
                        likelihoodr[2*i][2*l][2*k][2*m] = L16
                        

                        #Now have everything for this cell, so spline and integrate
                        vals = np.array([L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15,L16])
                        x1 = np.array([lnepshere[i] , lnepsr[2*i+1], lnepsr[2*i], lnepsr[2*i+1], lnepsr[2*i+1], lnepsr[2*i], lnepsr[2*i], lnepsr[2*i+1], lnepsr[2*i],lnepsr[2*i+1], lnepsr[2*i], lnepsr[2*i+1], lnepsr[2*i+1], lnepsr[2*i], lnepsr[2*i], lnepsr[2*i+1], lnepsr[2*i] ])
                        x2 = np.array([qhere[l] , qr[2*l+1],qr[2*l+1],qr[2*l],qr[2*l+1],qr[2*l],qr[2*l+1],qr[2*l],qr[2*l], qr[2*l+1],qr[2*l+1],qr[2*l],qr[2*l+1],qr[2*l],qr[2*l+1],qr[2*l],qr[2*l]])
                        x3 = np.array([logVhere[k] , logVr[2*k+1],logVr[2*k+1],logVr[2*k+1],logVr[2*k],logVr[2*k+1],logVr[2*k],logVr[2*k],logVr[2*k], logVr[2*k+1],logVr[2*k+1],logVr[2*k+1],logVr[2*k],logVr[2*k+1],logVr[2*k],logVr[2*k],logVr[2*k]])
                        x4 = np.array([nhere[m], nr[2*m+1], nr[2*m+1], nr[2*m+1], nr[2*m+1], nr[2*m+1], nr[2*m+1], nr[2*m+1], nr[2*m+1], nr[2*m], nr[2*m], nr[2*m], nr[2*m], nr[2*m], nr[2*m], nr[2*m], nr[2*m]])
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
                        x1 = np.linspace(lnepsr[2*i],lnepsr[2*i+1],NN)
                        x2 = np.linspace(qr[2*l],qr[2*l+1],NN)
                        x3 = np.linspace(logVr[2*k],logVr[2*k+1],NN)
                        x4 = np.linspace(nr[2*m],nr[2*m+1],NN)
                        
                        
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
                        
                        Eprime += res/((lneps[-1]-lneps[0])*(q[-1]-q[0])*(logV[-1]-logV[0])*(n[-1]-n[0])) #s_over_n#exp(-.5*chisq)
                        Log_new += res_log/((lneps[-1]-lneps[0])*(q[-1]-q[0])*(logV[-1]-logV[0])*(n[-1]-n[0])) 
                        numpoints += 1

                        


                        if any(y > .01 for y in vals[1:]):
                            if i == 0 and ifirst==False:
                                print('Problem? i = '+str(i))
                                ifirst = True
                            if l == 0 and lfirst==False:
                                print('Problem? l = '+str(l))
                                lfirst = True
                            if k == 0 and kfirst==False:
                                print('Problem? k = '+str(k))
                                kfirst = True
                            if m == 0 and mfirst==False:
                                print('Problem? m = '+str(m))
                                mfirst = True
                            if i == int(2**(1+z)*start-1) and ilast==False:
                                print('Problem? i = '+str(i))
                                ilast = True
                            if l == int(2**(1+z)*start-1) and llast==False:
                                print('Problem? l = '+str(l))
                                llast = True
                            if k == int(2**(1+z)*start-1) and klast==False:
                                print('Problem? k = '+str(k))
                                klast = True
                            if m == int(2**(1+z)*start-1) and mlast==False:
                                print('Problem? m = '+str(m))
                                mlast = True

                        if np.any(vals > Likemax): 
                            Likemax = np.max(vals)
                            Maxloc = [lnepsr[i],qr[l],logVr[k],nr[m]]
                        

                else:
                    for m in range(int(2**z*start)):

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


    dv = logVr[1]-logVr[0];dq = qr[1]-qr[0];dn = nr[1]-nr[0];deps = lnepsr[1]-lnepsr[0]

    print('Eprime is '+str(Eprime))
    print('numpoints is '+str(numpoints))
    print('summation_else = '+str(summation_else))
    print('L_lnL / N = '+str(Log_new))
    indexes = np.unravel_index(np.argmax(likelihoodr, axis=None), likelihoodr.shape)

    print('delta theta cell = '+str(dv*dq*dn*deps))
    print('delta theta code = '+str((logV[-1]-logV[0])*(q[-1]-q[0])*(lneps[-1]-lneps[0])*(n[-1]-n[0])))
    print('Max L is '+str(Likemax)+', at '+str(Maxloc))
    print('BETTER: Lmax = '+str(np.max(likelihoodr))+', at '+str(indexes))

    like_prev = likelihoodr
    print('z = '+str(z)+' took '+str(time.time() - begin))
