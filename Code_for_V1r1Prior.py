#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 00:00:30 2021

@author: joe
"""

import numpy as np
from math import *
from matplotlib import tri
from trispec_nonmin_Bessel import output_bessel
from scipy.interpolate import interpn
from scipy.interpolate import griddata
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
c = np.linspace(0,2,dim)#2,dim)
#COULD FIX AS??
d = np.linspace(.01,.02,dim)#.2,dim)
lnv = np.linspace(-23,-20,dim)#-25,-15,dim)
n = np.linspace(0,.5,dim)#7,dim)
print('d in '+str([d[0],d[-1]]))
print('n in '+str([n[0],n[-1]]))
print('lnv in '+str([lnv[0],lnv[-1]]))
print('c in '+str([c[0],c[-1]]))


L_lnL = 0
Likemax = 0
print('PRIORS ARE NOW PLANCK 1 SIGMA (could do for only NG?)')
Asmax = log10_10As+np.sqrt(2.8851542e-04)#3.25
Asmin = log10_10As-np.sqrt(2.8851542e-04)#2.85
fNL_lim_max = fNL_bestfit+5.1#1e2
fNL_lim_min = fNL_bestfit-5.1
gNL_lim_max = gNL_bestfit+6.5e4#1e2
gNL_lim_min = gNL_bestfit-6.5e4
alphaS_max = alpha+np.sqrt(4.5186311e-05)
alphaS_min = alpha-np.sqrt(4.5186311e-05)
numinside = 0
numoutside = 0

nhere = [n[int(size/start + i*dim/start)] for i in range(start)]
lnvhere = [lnv[int(size/start + i*dim/start)] for i in range(start)]
dhere = [d[int(size/start + i*dim/start)] for i in range(start)]
chere = [c[int(size/start + i*dim/start)] for i in range(start)]
ln1010As = np.zeros([start,start,start,start])
fNLarr = np.zeros([start,start,start,start])
gNLarr = np.zeros([start,start,start,start])
for i in range(len(dhere)):
    begin=time.time()
    print(i)
    for j in range(len(nhere)): 
        print('j = '+str(j))
        bhere = get_b(nhere[j])
        ns_th = ns; alphas_th = ekphase(nhere[j],bhere)[1]
        c_ek = sqrt(2* 3*(60+1)**nhere[j])
        for k in range(len(lnvhere)):
           # print('k = '+str(k))
            for l in range(len(chere)):
               # print('l = '+str(l))
                fNL_th,gNL_th,output_logAs,tau = output_bessel(bhere,dhere[i],chere[l],lnvhere[k],1,True)


                log10_10As_th = log(1e10*exp(output_logAs+ log((tau*kval/2)**(ns-1))))#log((kval*tau/2)**(-2*c_ek*(bhere-c_ek)/(c_ek**2-2)))))
        
                ln1010As[i][j][k][l] = log10_10As_th
                fNLarr[i][j][k][l] =fNL_th
                gNLarr[i][j][k][l] =gNL_th
                
                if  log10_10As_th >= Asmax or log10_10As_th <= Asmin or fNL_th>=fNL_lim_max or fNL_th<= fNL_lim_min or gNL_th>=gNL_lim_max or gNL_th<= gNL_lim_min or alphas_th>=alphaS_max or alphas_th<=alphaS_min:#abs(fNL_th) >=fNL_lim or abs(gNL_th) >= 1e6 or abs(alphas_th)>0.1:
                    numoutside += 1
                else:
                    numinside += 1
                    if i == start-1 and j>0:
                        print('Hits at i='+str(i))
                    if i == 0 and j>0:
                        print('Hits at i='+str(i))
                    if j == start-1 and j>0:
                        print('Hits at j='+str(j))
                    if k == 0 and j>0:
                        print('Hits at k='+str(k))
                    if k == start-1 and j>0:
                        print('Hits at k='+str(k))
                    if l == 0 and j>0:
                        print('Hits at l='+str(l))
                    if l == start-1 and j>0:
                        print('Hits at l='+str(l))
                    
                        #print('Maybe hits at '+str(i)+', '+str(j)+', '+str(k)+', '+str(l))

    print('This i took '+str(time.time() - begin))
print('So far, in = '+str(numinside)+', and out = '+str(numoutside))
As_prev = ln1010As
As_thresh = .1
print('I also shrunk As_thresh to 0.1')
for q in range(3): 
    begin = time.time()
    lnvr = [lnv[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))] 
    nr = [n[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))]
    dr = [d[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))]
    cr = [c[int(size/start/(2**(q+1))*(1 + 2*i))] for i in range(int(2**(4+q)))]

    Asr = np.zeros([len(dr),len(nr),len(lnvr),len(cr)])
    ifound = False
    lfound = False
    kfound = False
    mfound = False
    k0found = False
    for i in range(int(2**(1+q)*start)):
        if i%10==0:
            print('At i = '+str(i))

        for l in range(int(2**(1+q)*start)):

            ns_th = ns
            bhere = get_b(nr[l])
            c_ek = sqrt(2* 3*(60+1)**nr[l])

            ns_th = ekphase(nr[l],bhere)[0]
            alphas_th = ekphase(nr[l],bhere)[1]
            for k in range(int(2**(1+q)*start)):
                for m in range(int(2**(1+q)*start)):


                    if abs(As_prev[int(i/2)][int(l/2)][int(k/2)][int(m/2)]-log10_10As) < As_thresh:

                        fNL_th,gNL_th,logAs,tau = output_bessel(bhere,dr[i],cr[m],lnvr[k],1,True)
                        log10_10As_th = log(1e10*exp(logAs+ log((tau*kval/2)**(ns-1))))#log((kval*tau/2)**(-2*c_ek*(bhere-c_ek)/(c_ek**2-2)))))
                        Asr[i][l][k][m] = log10_10As_th
                        if log10_10As_th >= Asmax or log10_10As_th <= Asmin or fNL_th>=fNL_lim_max or fNL_th<= fNL_lim_min or gNL_th>=gNL_lim_max or gNL_th<= gNL_lim_min or alphas_th>=alphaS_max or alphas_th<=alphaS_min:#abs(fNL_th) >= 1e2 or abs(gNL_th) >= 1e6 or log10_10As_th>=Asmax or log10_10As_th<=Asmin or abs(alphas_th)>0.1:
                            numoutside += 1
                        else:
                            numinside += 1
                            #if i == int(2**(1+q)*start-1) or l == int(2**(1+q)*start-1) or m == int(2**(1+q)*start-1) or k ==0:
                            #    print('Problem? '+str(i)+', '+str(l)+', '+str(k)+', '+str(m))
                            if i == int(2**(1+q)*start-1) and ifound==False:
                                print('Problem? i = '+str(i))
                                ifound = True
                            if l == int(2**(1+q)*start-1) and lfound==False:
                                print('Problem? l = '+str(l))
                                lfound = True
                            if k == int(2**(1+q)*start-1) and kfound==False:
                                print('Problem? k = '+str(k))
                                kfound = True
                            if m == int(2**(1+q)*start-1) and mfound==False:
                                print('Problem? m = '+str(m))
                                mfound = True
                            if k == 0 and k0found==False:
                                print('Problem? k = '+str(k))
                                k0found = True


                        
                        


                    else:

                        numoutside += 1
                        Asr[i][l][k][m] = As_prev[int(i/2)][int(l/2)][int(k/2)][int(m/2)]


    print('Numin = '+str(numinside))
    print('Numout = '+str(numoutside))
    print('Ratio = '+str(numinside/(numinside + numoutside)))


    As_prev = Asr
    print('q = '+str(q)+' took '+str(time.time() - begin))