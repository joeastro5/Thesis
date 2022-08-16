#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 01:19:48 2021

@author: joe
"""

import numpy as np
from math import *
from matplotlib import tri
from Integ_deltasUnstable import new_result
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


def ekphase(n,N = 60):
    this = 2/(ns-1+n/(N+1))/(N+1)**n #This makes ns = ns_Planck
    eps_ek = this*(N+1)**n
    epsN = n*eps_ek/(N+1)
    epsNN = n*(n-1)*eps_ek/(N+1)**2
    c = sqrt(2*eps_ek) 
    nS = 1.+2/eps_ek - epsN/eps_ek
    alphaS = 2*epsN/eps_ek**2 + (epsNN/eps_ek - (epsN/eps_ek)**2) 
    return [nS,alphaS,c]


start = 8
size = 256
dim = 2*size

lneps = np.linspace(log(9),log(4000),dim)
n = np.linspace(2.23,167.1,dim)
logV = np.linspace(-60,-8,dim)
q = np.linspace(0,1.25,dim)

lnepshere = [lneps[int(size/start + i*dim/start)] for i in range(start)]
qhere = [q[int(size/start + i*dim/start)] for i in range(start)]
logVhere = [logV[int(size/start + i*dim/start)] for i in range(start)]
nhere = [n[int(size/start + i*dim/start)] for i in range(start)]

L_lnL = 0
Likemax = 0
Asmax = 3.25
Asmin = 2.85
numinside = 0
numoutside = 0

ln1010As = np.zeros([start,start,start,start])

ifirst = False
kfirst = False
ilast = False
jfirst = False
jlast = False
klast = False
llast = False
lfirst = False

for i in range(start):  #lneps
    print('i = '+str(i))
    for j in range(start): #q
        #print('j = '+str(j))
        for k in range(start): #logV
            fNL,gNL,logAs,tau = new_result(exp(lnepshere[i]),qhere[j],logVhere[k])

            for l in range(start):  #n
                
                #b = get_b(nhere[l],exp(lnepshere[i]))
               # eps_ek = exp(lnepshere[i])*(61)**nhere[l]
               # c_ek = sqrt(2*eps_ek)
                alpha_th = ekphase(nhere[l])[1] 
                ns_th = ekphase(nhere[l])[0] #ns
                log10_10As_th = log(1e10*exp(logAs+ log((tau*kval/2)**(ns-1))))

                    
                ln1010As[i][j][k][l] = log10_10As_th

                
                if  log10_10As_th >= Asmax or log10_10As_th <= Asmin or abs(fNL) >=1e2 or abs(gNL) >= 1e6 or abs(alpha_th)>0.1:
                    numoutside += 1
                else:
                    numinside += 1
                    if i == start-1 or i == 0 or k == start-1 or k==0 or l ==start-1 or l==0 or j ==start-1 or j==0:
                        print('Maybe hits at '+str(i)+', '+str(j)+', '+str(k)+', '+str(l))


print('So far, in = '+str(numinside)+', and out = '+str(numoutside))
As_prev = ln1010As
As_thresh = .3
for z in range(4): 
    begin = time.time()
    lnepsr = [lneps[int(size/start/(2**(z+1))*(1 + 2*i))] for i in range(int(2**(4+z)))] 
    nr = [n[int(size/start/(2**(z+1))*(1 + 2*i))] for i in range(int(2**(4+z)))]
    logVr = [logV[int(size/start/(2**(z+1))*(1 + 2*i))] for i in range(int(2**(4+z)))]
    qr = [q[int(size/start/(2**(z+1))*(1 + 2*i))] for i in range(int(2**(4+z)))]

    Asr = np.zeros([len(lnepsr),len(qr),len(logVr),len(nr)])
    
    
    kfirst = False
    ilast = False
    lfirst = False
    llast = False
    ifirst = False
    klast = False
    mlast = False
    mfirst = False
    
    for i in range(int(2**(1+z)*start)):#lneps
        #print('i = '+str(i))
        if i%10==0:
            print('At i = '+str(i))#+' , max like_ref is '+str(np.max(likelihoodr)))

        for l in range(int(2**(1+z)*start)):#q
            #print('l = '+str(l))
            
                        
            for k in range(int(2**(1+z)*start)):#lnV
                #print('k = '+str(k))
            
                if any(abs(y-log10_10As) < As_thresh for y in As_prev[int(i/2)][int(l/2)][int(k/2)]):
                    fNL_th,gNL_th,logAs,tau = new_result(exp(lnepsr[i]),qr[l],logVr[k])
                    for m in range(int(2**(1+z)*start)):#n
                        #print('m = '+str(m))


                    
                        ns_th = ekphase(nr[m])[0]#ns
                        
                       # eps_ek= exp(lnepsr[i])*(61)**nr[m]
                       # c_ek = sqrt(2*eps_ek)
                        
                        alphas_th = ekphase(nr[m])[1]

                        
                        log10_10As_th = log(1e10*exp(logAs+ log((kval*tau/2)**(ns-1))))


                        Asr[i][l][k][m] = log10_10As_th
                        if abs(fNL_th) >= 1e2 or abs(gNL_th) >= 1e6 or log10_10As_th>=Asmax or log10_10As_th<=Asmin or abs(alphas_th)>0.1:
                            numoutside += 1
                        else:
                            numinside += 1
                            #if i == int(2**(1+q)*start-1) or l == int(2**(1+q)*start-1) or m == int(2**(1+q)*start-1) or k ==0:
                            #    print('Problem? '+str(i)+', '+str(l)+', '+str(k)+', '+str(m))
                            if i == int(2**(1+z)*start-1) and ilast==False:
                                print('Problem? i = '+str(i))
                                ilast = True
                            if i == 0 and ifirst==False:
                                print('Problem? i = '+str(i))
                                ifirst = True
                            if m == int(2**(1+z)*start-1) and mlast==False:
                                print('Problem? m = '+str(m))
                                mlast = True
                            if k == int(2**(1+z)*start-1) and klast==False:
                                print('Problem? k = '+str(k))
                                klast = True
                            
                            if k == 0 and kfirst==False:
                                print('Problem? k = '+str(k))
                                kfirst = True
                            if m == 0 and mfirst==False:
                                print('Problem? m = '+str(m))
                                mfirst = True
                            if l == 0 and lfirst==False:
                                print('Problem? l = '+str(l))
                                lfirst = True
                            if l == int(2**(1+z)*start-1) and llast==False:
                                print('Problem? l = '+str(l))
                                llast = True


                        
                        


                else:
                    for m in range(int(2**(1+z)*start)):

                        numoutside += 1
                        Asr[i][l][k][m] = As_prev[int(i/2)][int(l/2)][int(k/2)][int(m/2)]


    print('Numin = '+str(numinside))
    print('Numout = '+str(numoutside))
    print('Ratio = '+str(numinside/(numinside + numoutside)))


    As_prev = Asr
    print('z = '+str(z)+' took '+str(time.time() - begin))