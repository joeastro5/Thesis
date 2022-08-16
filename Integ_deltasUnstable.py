#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:48:37 2021

@author: joe
"""

from matplotlib import pyplot as plt  
from math import * 
import cmath
import numpy as np 
import scipy.integrate as integ
import sympy as sym
from scipy.integrate import solve_ivp
import sys 
from sympy import Piecewise

import time



#NOTE could do fields and params as PyTransport files do.
#Note we need sym.[exp or sqrt], or else it tries to use math.[].


  
#Evolve background backwards in time to get ICs
#(I do this first because we need a(tstart) for delta_S ICs)
def prebounce_background(t,y): #y is the vector [phi,phidot]
    phiIC, phidotIC = y #,HIC,aIC = y
    kfunc = 1.-2/(1.+0.5*phiIC**2)**2 
    qfunc = q/(1.+0.5*phiIC**2)**2
    
    rhoIC = 0.5*kfunc*phidotIC**2+0.75*qfunc*phidotIC**4+v(phiIC,0.,kappa3)
    HIC = -sqrt(abs(rhoIC)/3)
 
    pphi = (4.*phiIC/(1.+0.5*phiIC**2)**3)*.5*phidotIC**2-0.25*2*q*phiIC/(1.+0.5*phiIC**2)**3*phidotIC**4
    px = kfunc+qfunc*phidotIC**2
    pxx = 2.*qfunc
    pxphi = 4.*phiIC/(1.+0.5*phiIC**2)**3-2*q*phiIC/(1.+0.5*phiIC**2)**3*phidotIC**2
    vp = vphi(phiIC,0.,kappa3)
    phidotdotIC = (pphi-px*3*HIC*phidotIC-vp-pxphi*phidotIC**2)/(px+pxx*phidotIC**2)
                  
    dydt = [phidotIC, phidotdotIC]
    return dydt

################################################

def prebounce_pert(t,y):  
    phiS,phidotS = y[0:2]
    delta_sL_full = y[2:3]
    delta_sLdot_full = y[3:4]
    N = y[-1]
    kfunc = 1.-2/(1.+0.5*phiS**2)**2 
    qfunc = q/(1.+0.5*phiS**2)**2
    vcc = vchichi(phiS,0.,kappa3)
    v_ss = vcc  

    rhoS = 0.5*kfunc*phidotS**2+0.75*qfunc*phidotS**4+v(phiS,0.,kappa3)
    HS = -sqrt(abs(rhoS)/3.)
    Ndot = HS
    aS = exp(y[-1])
     
    pphi = (4.*phiS/(1.+0.5*phiS**2)**3)*.5*phidotS**2-0.25*2*q*phiS/(1.+0.5*phiS**2)**3*phidotS**4
    px = kfunc+qfunc*phidotS**2
    pxx = 2.*qfunc
    pxphi = 4.*phiS/(1.+0.5*phiS**2)**3-2*q*phiS/(1.+0.5*phiS**2)**3*phidotS**2
    vp = vphi(phiS,0.,kappa3)

    phidotdotS = (pphi-px*3*HS*phidotS-vp-pxphi*phidotS**2)/(px+pxx*phidotS**2)
        
    adotdot = aS*(-.5*kfunc*phidotS**2-.5*qfunc*phidotS**4+HS**2)
    delta_sLdotdot = -3*delta_sLdot_full*HS - v_ss*delta_sL_full
    first = np.append(phidotS,phidotdotS);second = np.append(delta_sLdot_full,delta_sLdotdot)
    this = np.append(first,second)
    dydt = np.append(this,Ndot)
    return dydt
    

#plot with, for example, plt.plot(solIC_pert.t,solIC_pert.y[0]) --> phi


def postbounce(t,y):
    phi,phidot,chi,chidot,delta_sL_full,delta_sLdot_full,delta_s2_full,delta_s2dot_full,delta_s3_full,delta_s3dot_full,zetaL_full,zeta2_full,zeta3_full,N = y  
    
    kfunc = 1.-2/(1.+0.5*phi**2)**2 
    qfunc = q/(1.+0.5*phi**2)**2

    sigdot = sqrt(phidot**2+chidot**2)
   
    vp = vphi(phi,chi,kappa3);vpp = vphiphi(phi,chi,kappa3);vppp = vphiphiphi(phi,chi,kappa3);vpppp = 0#vphiphiphiphi(phi,chi,kappa3)
    vc = vchi(phi,chi,kappa3);vcc = vchichi(phi,chi,kappa3);vccc = vchichichi(phi,chi,kappa3);vcccc = vchichichichi(phi,chi,kappa3)
    vpc = vphichi(phi,chi,kappa3);vpcc = vphichichi(phi,chi,kappa3);vppc = vphiphichi(phi,chi,kappa3);pot = v(phi,chi,kappa3)
    vpppc = vphiphiphichi(phi,chi,kappa3); vppcc = vphiphichichi(phi,chi,kappa3);vpccc = vphichichichi(phi,chi,kappa3)


    Vsigsig = (phidot**2*vpp+chidot**2*vcc+2*phidot*chidot*vpc)/sigdot**2
    Vsssig = (chidot**2*phidot*vppp - 2*chidot*phidot**2*vppc + chidot**3*vppc + phidot**3*vpcc - 2*phidot*chidot**2*vpcc + phidot**2*chidot*vccc)/sigdot**3
    
    #print('phi = '+str(phi))
    #print('chi = '+str(chi))
    #print('vpppp = '+str(vpppp))
    
    
    rho = 0.5*kfunc*phidot**2+0.75*qfunc*phidot**4+0.5*chidot**2+pot

    H = sqrt(abs(rho)/3.)
    a = exp(N)
  
    pphi = (4.*phi/(1.+0.5*phi**2)**3)*.5*phidot**2-0.25*2*q*phi/(1.+0.5*phi**2)**3*phidot**4
    px = kfunc+qfunc*phidot**2
    pxx = 2.*qfunc
    pxphi = 4.*phi/(1.+0.5*phi**2)**3-2*q*phi/(1.+0.5*phi**2)**3*phidot**2
    phidotdot = (pphi-px*3*H*phidot-vp-pxphi*phidot**2)/(px+pxx*phidot**2)
    
    chidotdot=-3*H*chidot-vc
    
    sigdotdot = phidot/sigdot*phidotdot-3*H*chidot**2/sigdot-chidot/sigdot*vc#-3*H*sigdot-(phidot*Vphi(phi,chi)+chidot*Vchi(phi,chi))/sigdot
    Vsdot = (3*H*chidot+vc)/sigdot*vp+chidot*sigdotdot/sigdot**2*vp-chidot/sigdot*(vpp*phidot+vpc*chidot)+phidotdot/sigdot*vc-phidot*sigdotdot/sigdot**2*vc+phidot/sigdot*(vcc*chidot+vpc*phidot)
     
    #AVOID CALLING FUNC FOR SPEED?
      
   
    #if v_rep(phi,chi) < cutoff and abs(chi) > 0.5:#(5e-18)/2e-9*Vo and abs(chi) > 0.5:#if v_rep(phi,chi) < 1e-19 and abs(chi) > 0.5:
    
    
    #Set these to zero by default, and make nonzero
    #if we are sometime before the end of the repulsive phase
    thdot = 0.
    Vs = 0.
    Vsig = 0.
    v_ss = 0.
    thdotdot =  0.
    vsss =0.
    vssss=0.
    Vssig = 0.
        #print('REPULSIVE FORCED OFF, at t = '+str(t))
    
 #   if vrep(phi,chi) >= cutoff:#.01*(5e-19)/2e-9*Vo:#else:
    Vs = -chidot*vp/sigdot+phidot*vc/sigdot
    thdot = -Vs/sigdot
    Vsig = (phidot*vp+chidot*vc)/sigdot
    v_ss = (chidot**2*vpp+phidot**2*vcc-2.*chidot*phidot*vpc)/sigdot**2
    thdotdot = -Vsdot/sigdot+Vs/sigdot**2*sigdotdot
    Vssig = (phidot*chidot*(vcc-vpp)+(phidot**2-chidot**2)*vpc)/sigdot**2
    vsss = -(chidot/sigdot)**3*vppp+3*chidot**2*phidot/sigdot**3*vppc-3*chidot*phidot**2/sigdot**3*vpcc+(phidot/sigdot)**3*vccc
    vssss = (chidot/sigdot)**4*vpppp-4*chidot*phidot**3/sigdot**4*vpccc+6*phidot**2*chidot**2/sigdot**4*vppcc-4*phidot*chidot**3/sigdot**4*vpppc+(phidot/sigdot)**4*vcccc
    
        #print('REPULSIVE STILL ON, at t = '+str(t))
    
    #It may be this next part is not needed
 #   elif vrep(phi,chi) < cutoff and abs(chi) < 0.5:#else:
 #       Vs = -chidot*vp/sigdot+phidot*vc/sigdot
 #       thdot = -Vs/sigdot
 #       Vsig = (phidot*vp+chidot*vc)/sigdot
 #       v_ss = (chidot**2*vpp+phidot**2*vcc-2.*chidot*phidot*vpc)/sigdot**2
 #       thdotdot = -Vsdot/sigdot+Vs/sigdot**2*sigdotdot
 #       Vssig = (phidot*chidot*(vcc-vpp)+(phidot**2-chidot**2)*vpc)/sigdot**2
 #       vsss = -(chidot/sigdot)**3*vppp+3*chidot**2*phidot/sigdot**3*vppc-3*chidot*phidot**2/sigdot**3*vpcc+(phidot/sigdot)**3*vccc
 #       vssss = (chidot/sigdot)**4*vpppp-4*chidot*phidot**3/sigdot**4*vpccc+6*phidot**2*chidot**2/sigdot**4*vppcc-4*phidot*chidot**3/sigdot**4*vpppc+(phidot/sigdot)**4*vcccc
 #       t_endrep = t
        #print('REPULSIVE STILL ON, at t = '+str(t))
   # else:
    #    print('REPULSIVE FORCED OFF, at t = '+str(t))
    #print('here, '+str(v_rep(phi,chi)))
    #print('chi = '+str(chi))
    adotdot = a*(-0.5*kfunc*phidot**2-.5*qfunc*phidot**4-0.5*chidot**2+H**2)#y[-1]*(-.5*kfunc*phidot**2-.5*qfunc*phidot**4-.5*chidot**2+y[-2]**2)
    
    #--------------------------------
    
    
    delta_sLdotdot=-3*delta_sLdot_full*H+(-v_ss-3*thdot**2
        )*delta_sL_full

    delta_s2dotdot= -3*delta_s2dot_full*H+(-v_ss-3*thdot**2
        )*delta_s2_full-thdot/sigdot*delta_sLdot_full**2-2./sigdot*(thdotdot
        +Vsig*thdot/sigdot-1.5*H*thdot)*delta_sLdot_full*delta_sL_full+(
            -.5*vsss+5*v_ss*thdot/sigdot+9*thdot**3/sigdot)*delta_sL_full**2

    
    delta_s3dotdot = -3*H*delta_s3dot_full - (v_ss + 3*thdot**2)*delta_s3_full - 2*thdot/sigdot*delta_s2dot_full*delta_sLdot_full - (
        2*thdotdot/sigdot+2*thdot*Vsig/sigdot**2-3*H*thdot/sigdot)*(delta_s2dot_full*delta_sL_full+delta_s2_full*delta_sLdot_full)-(
            vsss-10*thdot*v_ss/sigdot-18*thdot**3/sigdot)*delta_s2_full*delta_sL_full - Vsig/sigdot**3*delta_sLdot_full**3 - (
                Vsigsig/sigdot**2+3*Vsig**2/sigdot**4+3*H*Vsig/sigdot**3-2*v_ss/sigdot**2-6*thdot**2/sigdot**2)*delta_sLdot_full**2*delta_sL_full-(
                    -10*thdot*thdotdot/sigdot**2-1.5/sigdot*Vsssig-5*Vsig*v_ss/sigdot**3-7*thdot**2*Vsig/sigdot**3-3*H*v_ss/sigdot**2+14*H*thdot**2/sigdot**2
                    )*delta_sLdot_full*delta_sL_full**2-(vssss/6-7./3*thdot/sigdot*vsss+2*v_ss**2/sigdot**2+21*thdot**2*v_ss/sigdot**2+27*thdot**4/sigdot**2)*delta_sL_full**3
                                   
    zetaLdot = -2.*H/sigdot*thdot*delta_sL_full
   
    zeta2dot = -2.*H/sigdot*thdot*delta_s2_full+H/sigdot**2*((v_ss+4*thdot**2
        )*delta_sL_full**2-Vsig/sigdot*delta_sL_full*delta_sLdot_full) 
    #^Note typo in Fertig; Can also do this to third order, see https://arxiv.org/pdf/0909.2558.pdf
    zeta3dot = -2.*H/sigdot*thdot*delta_s3_full+H/sigdot**2*(-Vsig/sigdot*(delta_sLdot_full*delta_s2_full+delta_sL_full*delta_s2dot_full)-thdot/3*Vsig/sigdot**2*delta_sL_full**2*delta_sLdot_full+2*v_ss*delta_sL_full*delta_s2_full-Vssig/sigdot*delta_sL_full**2*delta_sLdot_full+vsss/3*delta_sL_full**3
                                                        )+8*H/sigdot**4*(Vs**2*delta_sL_full*delta_s2_full-Vs*Vsig/(2*sigdot)*delta_sL_full**2*delta_sLdot_full+.5*Vs*v_ss*delta_sL_full**3)+8*H/sigdot**6*Vs**3*delta_sL_full**3
    
    Ndot = H
    
    
    if rho <= 0. and t > sqrt(q):
        #print('Density became zero at t = '+str(t)+'! Exiting')
       # print('v_rep = '+str(v_rep(phi,chi)))
        #print(chi)
        return #THIS MAKES t_endrep THE LAST TIME!
               #^I fixed this with 'set these to zero...' above and the elif 
               #But I am not sure why with just the original if and else I would up calling the else, resuling in t_endrep being the final time

    #print('At t = '+str(t)+' and vr = '+str(v_rep(phi,chi))+' and phi = '+str(phi)+' and chi = '+str(chi))
    return phidot,phidotdot,chidot,chidotdot,delta_sLdot_full,delta_sLdotdot,delta_s2dot_full,delta_s2dotdot,delta_s3dot_full,delta_s3dotdot,zetaLdot,zeta2dot,zeta3dot,Ndot#,thdot #dydt


#plot with, for example, plt.plot(solIC_pert.t,solIC_pert.phi)
hubble = []
sigmadot =[]
a_post = []
ds = []
#zldot = []
thd = []
#minusvs = []
#cdotvphi = []
#minuspdotvchi = []
#minusthreehpd = []
#minusvphi = []
#kterm = []
#qterm = []
#ekpot = []
reppot = []
term1 = [];term2 = [];term3 = [];term4 = [];
#chikin = []
#pdd = [];denom=[];first=[];second=[];third=[];fourth=[]
#for i in range(len(sol_post.t)):
#    p = sol_post.y[0][i]
#    pdot = sol_post.y[1][i]
#    c = sol_post.y[2][i]
#    cdot = sol_post.y[3][i]
#    kfunc = 1.-2/(1.+0.5*p**2)**2 
#    qfunc = q/(1.+0.5*p**2)**2
#    hubble.append(sqrt(abs(0.5*kfunc*pdot**2+.5*cdot**2+0.75*qfunc*pdot**4+v(p,c,kappa3))/3))
#    term1.append(0.5*kfunc*pdot**2)
#    term2.append(.5*cdot**2)
#    term3.append(0.75*qfunc*pdot**4)
#    term4.append(v(p,c))
 #   kterm.append(.5*pdot**2*kfunc)
 #   qterm.append(qfunc*.75*pdot**4)
 #   ekpot.append(v(p,c)-vrep(p,c))
#    reppot.append(vrep(p,c))
 #   chikin.append(.5*cdot**2)
 #   minusthreehpd.append(-3*hubble[i]*pdot)
 #   minusvphi.append(-vphi(p,c))
#    sigmadot.append(sqrt(pdot**2+cdot**2))
  #  a_post.append(exp(sol_post.y[-1][i]))
  #  ds.append(sol_post.y[4][i]/a_post[i])
    
 #   pphi = (4.*p/(1.+0.5*p**2)**3)*.5*pdot**2-0.25*2*q*p/(1.+0.5*p**2)**3*pdot**4
 #   px = kfunc+qfunc*pdot**2
 #   pxx = 2.*qfunc
 #   pxphi = 4.*p/(1.+0.5*p**2)**3-2*q*p/(1.+0.5*p**2)**3*pdot**2
 #   phidotdot = (pphi-px*3*hubble[i]*pdot-vphi(p,c)-pxphi*pdot**2)/(px+pxx*pdot**2)
 #   pdd.append(phidotdot)
 #   denom.append(px+pxx*pdot**2)
 #   first.append(pphi)
 #   second.append(-px*3*hubble[i]*pdot)
#    third.append(-vphi(p,c))
#    fourth.append(-pxphi*pdot**2)
#    minusvs.append(cdot*vphi(p,c)/sigmadot[i]-pdot*vchi(p,c)/sigmadot[i])
#    cdotvphi.append(cdot*vphi(p,c))
#    minuspdotvchi.append(-pdot*vchi(p,c))
#    thd.append(cdot*vphi(p,c)/sigmadot[i]**2-pdot*vchi(p,c)/sigmadot[i]**2)
#    zldot.append(-2.*hubble[i]/sigmadot[i]*thd[i]*ds[i])
#Final output: ns, As, etc AT k = k_pivot!!!

#Need to decide when conv ends, eg when zeta_dot has gotten small enough
#Sp_t = np.linspace(sol_post.t[0],sol_post.t[-1],1000)
#Sp_zetaL_dot = UnivariateSpline(sol_post.t,sol_post.y[4+4*len(karray):4+5*len(karray)],s=0).derivative()(Sp_t)
#Find time when zetaL_dot is five times larger than the final one
#for i in range(1,1000):
#    if Sp_zetaL_dot[-i-1] >= 5.*Sp_zetaL_dot[-1]:
        #print(i)
#        break
#^This is when we want to evaluate fNL
#Sp_zetaL = UnivariateSpline(sol_post.t,sol_post.y[4+4*len(karray):4+5*len(karray)],s=0)(Sp_t)
#Sp_zeta2 = UnivariateSpline(sol_post.t,sol_post.y[4+5*len(karray):4+6*len(karray)],s=0)(Sp_t)
#Sp_zetaL_evaluate = Sp_zetaL[-i+1]
#Sp_zeta2_evaluate = Sp_zeta2[-i+1]
#fNL = 5./3*Sp_zeta2_evaluate/Sp_zetaL_evaluate**2


#could get ns from analytic, but should verify zeta scales in the correct way at late times

def new_result(eps,q_input,logV):#,thr,logvr):
    global q, vrep,v,vphi,vphiphi,vphiphiphi,vphiphiphiphi,vchi,vchichi,vchichichi,vchichichichi,vphichi,vphichichi,vphiphichi,vphiphiphichi,vphiphichichi,vphichichichi,kappa3
    r_tol = 1e-8#6
    a_tol = 1e-8#6
    integtype = 'DOP853'#'RK45' #For non stiff, use ‘RK23’, ‘RK45’, or ‘DOP853’ 
    kappa3 = 0.
    Vo = exp(logV)
    
    thr = 1.
    logvr = -4.
    
    th_rep = thr*pi/10
    c = cos(th_rep)   #Parameters of repulsive potential
    b = sin(th_rep)
    q = q_input/(5*Vo)
    start = time.time()
#   k3 = 0.#float(sys.argv[2])

    phi, chi, k3 = sym.symbols('phi chi k3')
    #phiend = .992415568 #stable
    phiend = -.992415568 #unstable
    phipart = -2.*Vo/(sym.exp(sym.sqrt(2.*eps)*phi)+sym.exp(-sym.sqrt(2.*eps)*phi))
    vek = Piecewise((phipart*(1.+(eps*chi**2+k3/6*eps**1.5*chi**3)),phi > phiend),(phipart,phi <=phiend))#*(1+.5*eps*chi**2)#*(1.+1./(1.+sym.exp(-50*(phi-.9924)))*(.5*eps*chi**2+k3/6*eps**1.5*chi**3))#I think this -> was for the ek trispec paper?Piecewise((-Vo*exp(-sqrt(2.*epsilon)*phi),phi>= 1.3),(0.,True))
    vrep = 10**(logvr)*Vo*sym.exp(-5.*(b*phi-c*chi+2)**2)#exp(float(sys.argv[5]))*sym.exp(-5.*(b*phi-c*chi+2)**2)#1e-4*Vo*sym.exp(-5.*(b*phi-c*chi+2)**2)
    
    v = vek+vrep
    vphi = sym.diff(v,phi)
    vphiphi = sym.diff(v,phi,2)
    vphiphiphi = sym.diff(v,phi,3)
    vphiphiphiphi = sym.diff(v,phi,4)
    #The first argument of diff can be a function I defined, just be sure to include its args, eg Vrep(phi,chi).
    vchi = sym.diff(v,chi)
    vphichi = sym.diff(v,phi,chi);vchichi = sym.diff(v,chi,chi)
    vphiphichi = sym.diff(v,phi,phi,chi);vphichichi = sym.diff(v,phi,chi,chi)
    vchichichi = sym.diff(v,chi,chi,chi)
    vchichichichi = sym.diff(v,chi,4)

    vphiphiphichi = sym.diff(v,phi,phi,phi,chi)
    vphiphichichi = sym.diff(v,phi,phi,chi,chi)
    vphichichichi = sym.diff(v,phi,chi,chi,chi)

    vrepphiphiphi = sym.diff(vrep,phi,3)
    vrepchi = sym.diff(vrep,chi)
    vrepphi = sym.diff(vrep,phi)
    vrepphiphi = sym.diff(vrep,phi,phi)
    vrepphichi = sym.diff(vrep,phi,chi)
    vrepchiphi = sym.diff(vrep,chi,phi)
    vrepchichi = sym.diff(vrep,chi,chi)
    vrepphiphichi = sym.diff(vrep,phi,phi,chi);vrepphichichi = sym.diff(vrep,phi,chi,chi)
    vrepchichichi = sym.diff(vrep,chi,chi,chi)

    #Then make them callable functions; note they are arrays, may have to make intofloats later
    v = sym.lambdify((phi,chi,k3),v)
    vrep = sym.lambdify((phi,chi),vrep)
    vrepphi =sym.lambdify((phi,chi),vrepphi);vrepchi =sym.lambdify((phi,chi),vrepchi) ;vrepphichi=sym.lambdify((phi,chi),vrepphichi)
    vphi = sym.lambdify((phi,chi,k3),vphi);vchi = sym.lambdify((phi,chi,k3),vchi)
    vphiphi = sym.lambdify((phi,chi,k3),vphiphi);vphichi = sym.lambdify((phi,chi,k3),vphichi);vchichi = sym.lambdify((phi,chi,k3),vchichi) 
    vphiphiphi = sym.lambdify((phi,chi,k3),vphiphiphi);vphiphichi = sym.lambdify((phi,chi,k3),vphiphichi)
    vphichichi = sym.lambdify((phi,chi,k3),vphichichi);vchichichi = sym.lambdify((phi,chi,k3),vchichichi)
    vphiphiphiphi = sym.lambdify((phi,chi,k3),vphiphiphiphi)
    vphiphiphichi = sym.lambdify((phi,chi,k3),vphiphiphichi)
    vphiphichichi = sym.lambdify((phi,chi,k3),vphiphichichi)
    vphichichichi = sym.lambdify((phi,chi,k3),vphichichichi)
    vchichichichi = sym.lambdify((phi,chi,k3),vchichichichi)

    vrepphiphi = sym.lambdify((phi,chi),vrepphiphi)
    vrepchichi = sym.lambdify((phi,chi),vrepchichi)

    vrepphiphiphi = sym.lambdify((phi,chi),vrepphiphiphi);vrepphiphichi = sym.lambdify((phi,chi),vrepphiphichi)
    vrepphichichi = sym.lambdify((phi,chi),vrepphichichi);vrepchichichi = sym.lambdify((phi,chi),vrepchichichi)
    end = time.time()
    sympytime = end - start
    start = time.time()
    
    #Required values of things at t = 0
    phiIC = [0.] 
    phidotIC = [-np.sqrt((1.+np.sqrt(1.+12*q*Vo))/(3*q))] #Enforces rho(0) = 0; Opposite sign in Fertig, I am following Lehners file

    #IC
    y0 = [phiIC[0], phidotIC[0]]#,0.,1.]
    #t = np.linspace(0., -1000000, 100) #Making third arg 1000 seems to be worse (bounce phase begins later)
    t = np.linspace(0., -2.5e7, 100)
    solIC_back = solve_ivp(prebounce_background, [t[0], t[-1]], y0, t_eval=t,rtol = r_tol,atol = a_tol,method = integtype)#odeint(prebounce_background, y0, t)
    #I demand this tolerance here and with the next function; if not, the forward and backward phi
    #and phidot disagree; becomes a bigger problem with eg eps=40 and Vo>=7e-9, phidot blows up.
  #  plt.figure()
  #  plt.plot(solIC_back.t,solIC_back.y[0])
  #  plt.show()
 #   print('sol = '+str(solIC_back.t[-1]))
    #Now, use result to get a,H? At least to get tau.
    HIC = []#[0.]
    aIC = []#[1.]
    density = []
    for i in range(len(solIC_back.t)):
        p = solIC_back.y[0][i]
        pdot = solIC_back.y[1][i]
        kfunc = 1.-2/(1.+0.5*p**2)**2 
        qfunc = q/(1.+0.5*p**2)**2
        density.append(0.5*kfunc*pdot**2+0.75*qfunc*pdot**4+v(p,0.,kappa3))
        HIC.append(-sqrt(abs(density[i])/3))
    
    NIC = integ.cumtrapz(HIC, solIC_back.t, 10,initial = 0.)
  #  plt.figure()
  #  plt.plot(solIC_back.t,NIC)
  #  plt.show()
  #  plt.figure()
  #  plt.plot(solIC_back.t,density)
  #  plt.title('Density, prebounce')
    
    arg = []
    for i in range(len(solIC_back.t)):
        aIC.append(exp(NIC[i]))
        arg.append(1./aIC[i])
    tau = integ.cumtrapz(arg , solIC_back.t, 10)#conformal time  
    
    delta_sL = 10**(-5)
    delta_sLdot = 10**(-5)/2.5e7

    
    first = np.append(solIC_back.y[0][-1], solIC_back.y[1][-1]); second = np.append(delta_sL,delta_sLdot)#;third = np.append(solIC_back[-1,2], solIC_back[-1,3])
    this = np.append(first,second)
    y0 = np.append(this,log(aIC[-1]))#third)
    #tnew = np.linspace(-1000000,0., 100)
    tnew = np.linspace(-2.5e7,0., 1000)
    solIC_pert = solve_ivp(prebounce_pert, [tnew[0], tnew[-1]], y0, t_eval=tnew,rtol=r_tol,atol = a_tol,method = integtype)#tnew,y0)

    delta_s2_init = 0.
    delta_s2dot_init = 0.
    delta_s3_init = 0.
    delta_s3dot_init = 0.
    zetaL_init =0.;zeta2_init = 0.;zeta3_init = 0.


    first = np.append(0., solIC_back.y[1,0]);second =np.append(0.,0.);
    third = np.append(solIC_pert.y[2][-1],solIC_pert.y[3][-1])#np.append(delta_SL[-1],delta_SLdot[-1])
    fourth = np.append(delta_s2_init,delta_s2dot_init);fifth = np.append(delta_s3_init,delta_s3dot_init);sixth = np.append(zetaL_init,zeta2_init);seventh = np.append(zeta3_init,0.)
    this = np.append(first,second);this2 = np.append(this,third);this3 = np.append(this2,fourth);this4 = np.append(this3,fifth);this5 = np.append(this4,sixth)

    y0 = np.append(this5,seventh)
    
    
    tlate =  1.5*sqrt(exp(14)/Vo) #I find this is large enough so the desired end time is smaller than it (the 1.5 is what should ensure it)      #500000000000000000
    tpost = np.linspace(0,tlate,10000)#Last number doesn't affect integration, just how many points are shown in plot
#    print('tlate = '+str(tlate))
    sol_post = solve_ivp(postbounce, [tpost[0], tpost[-1]], y0, t_eval=tpost,rtol = r_tol,atol = a_tol,method = integtype)
    hubble = []
    for i in range(len(sol_post.t)):
        p = sol_post.y[0][i]
        pdot = sol_post.y[1][i]
        c = sol_post.y[2][i]
        cdot = sol_post.y[3][i]
        kfunc = 1.-2/(1.+0.5*p**2)**2 
        qfunc = q/(1.+0.5*p**2)**2
        hubble.append(sqrt(abs(0.5*kfunc*pdot**2+.5*cdot**2+0.75*qfunc*pdot**4+v(p,c,kappa3))/3))
  #  plt.figure()
  #  plt.plot(sol_post.t,hubble)
  #  plt.title('hubble')
  #  plt.show()
#    plt.figure()
#    print('solpostt = '+str(sol_post.t[-1]))
    vr = []
    for i in range(len(sol_post.t)):
        vr.append(vrep(sol_post.y[0][i],sol_post.y[2][i]))
 #   plt.plot(sol_post.t,vr)
 #   plt.title(r'Repulsive potential')
    
#    print('sol_post.t[-1] = '+str(sol_post.t[-1]))
#    hubble = []
    
   # reppot = []
   # term1 = [];term2 = [];term3 = [];term4 = [];
 #   for i in range(len(sol_post.t)):
 #           p = sol_post.y[0][i]
 #           pdot = sol_post.y[1][i]
 #           c = sol_post.y[2][i]
 #           cdot = sol_post.y[3][i]
 #           kfunc = 1.-2/(1.+0.5*p**2)**2 
 #           qfunc = q/(1.+0.5*p**2)**2
 #           hubble.append(sqrt(abs(0.5*kfunc*pdot**2+.5*cdot**2+0.75*qfunc*pdot**4+v(p,c,kappa3))/3))
   # plt.figure()
   # plt.plot(sol_post.t,hubble)
   # plt.title('Hubble')
 #   plt.figure()
 #   plt.plot(sol_post.t,sol_post.y[13])
    
    
    vrmax = np.max(vr)
    vrmax_index = np.where(vr == vrmax)
 #   print('vrmax_index = '+str(vrmax_index))
    t_vrmax = sol_post.t[vrmax_index][0]
 #   print('t_vrmax = '+str(t_vrmax))
 #   print('Vr max is '+str(vrmax))
 #   print('t is '+str(np.shape(sol_post.t)))
    
    ind = np.where((vr <= 2.1e-5*vrmax) & (sol_post.t >= t_vrmax))
    #print('ind[0] = '+str(ind[0]))
    if len(ind[0]) > 1: 
   #     print('Happens')
        tend = sol_post.t[ind[0][0]]
    else:
        tend = tlate
 #   print('tend = '+str(tend))
    tpost = np.linspace(0,tend,10000)
    sol_post = solve_ivp(postbounce, [tpost[0], tpost[-1]], y0, t_eval=tpost,rtol = r_tol,atol = a_tol,method = integtype)


    nu = 1.5 #close enough
    amp = (-tau[-1])**(.5-nu)/sqrt(2) / 10**(-5)
    As = 1/(2*pi**2)*sol_post.y[10][-1]**2*amp**2#/amp**2
    tau_f = -tau[-1]
    lnAs = log(As)
    fNL = 5./3*sol_post.y[11][-1]/sol_post.y[10][-1]**2
    gNL = 25/9*sol_post.y[12][-1]/sol_post.y[10][-1]**3
    

    end = time.time()
    resttime = end - start
    
    timefull = np.append(solIC_pert.t,sol_post.t)
    dsfull = np.append(solIC_pert.y[2],sol_post.y[4])
    dsdotfull = np.append(solIC_pert.y[3],sol_post.y[5])
  #  plt.figure()
  #  plt.plot(timefull,dsfull,label = r'$\epsilon = $'+str(eps))
  #  plt.grid(b = True)
  #  plt.ylabel(r'$\delta s_L$')
  #  plt.title(r'$\delta s_L$')
    #plt.legend()
    #plt.title(r'$\delta s_i = $ '+str(delta_sL) + ', rtol and atol = ' +str(r_tol) + ', integ type is '+str(integtype))
    #plt.xlabel(r'$t$')
 
    
   # plt.figure()
  #  plt.plot(timefull,dsdotfull)
  #  plt.grid(b = True)
  #  plt.title(r'$\delta \dot{s}_L$,' )
  #  plt.xlabel(r'$t$')
    


    
  #  plt.figure()
  #  plt.plot(sol_post.t,sol_post.y[-1])
  #  plt.plot(solIC_pert.t,solIC_pert.y[-1])
  #  plt.plot(solIC_back.t,NIC,label='NIC')
  #  plt.legend()
  #  plt.title(r'$N$')
    scale = []
    for i in range(len(sol_post.t)):
        scale.append(exp(sol_post.y[-1][i]))
#    plt.figure()
#    plt.plot(sol_post.t,scale)
#    plt.title('a')
    
 #   plt.figure()
 #   plt.plot(sol_post.t,sol_post.y[0])
 #   plt.plot(solIC_pert.t,solIC_pert.y[0])
 #   plt.plot(solIC_back.t,solIC_back.y[0])
 #   plt.title(r'$\phi$')
 
 
    
 #   plt.figure()
 #   plt.plot(sol_post.t,sol_post.y[2])
 #   plt.title(r'$\chi$')
 #   print('this is '+str(exp(-20.0*(-0.475528258147577*sol_post.y[2][-1] + 0.154508497187474*sol_post.y[0][-1] + 1)**2)))
 #   print('and '+str(exp(-32.86335345031*sol_post.y[0][-1])))
#    plt.figure()
#    plt.plot(sol_post.t,sol_post.y[6])
#    plt.grid(b = True)
#    plt.title(r'$\delta s_2$')
#    plt.xlabel(r'$t$')
    
#    plt.figure()
#    plt.plot(sol_post.t,sol_post.y[8])
#    plt.grid(b = True)
#    plt.title(r'$\delta s_3$')
#    plt.xlabel(r'$t$')
    
#    plt.figure()
  #  plt.xlim(right=2e7)
#    plt.plot(sol_post.t,sol_post.y[10],label = r'$V_o = $'+str(Vo))# r'$5q V_o = $' + str(q_input))#label =  r'$ln V_o = $' + str(logV))#label = r'$\delta s_i = $ '+str(delta_sL))
#    plt.grid(b = True)
#    plt.title(r'Unstable potential, $\zeta_L$')#(r'$\zeta_L, V_o = $'+str(Vo)+ r', $5q V_o = $' + str(q_input)+r', $\epsilon = $'+str(eps))
#    plt.legend()
#    plt.xlabel(r'$t$')
    
#    plt.figure()
#    plt.plot(sol_post.t,sol_post.y[11],label = r'$V_o = $'+str(Vo))# r'$5q V_o = $' + str(q_input))#label =  r'$ln V_o = $' + str(logV))#label = r'$\delta s_i = $ '+str(delta_sL))
#    plt.grid(b = True)
#    plt.title(r'Unstable potential, $\zeta_2$')#(r'$\zeta_L, V_o = $'+str(Vo)+ r', $5q V_o = $' + str(q_input)+r', $\epsilon = $'+str(eps))
#    plt.legend()
#    plt.xlabel(r'$t$')
    
#    plt.figure()
#    plt.plot(sol_post.y[0],sol_post.y[2],label = 'Q = '+str(q_input))
#    plt.grid(b = True)
#    plt.title('phi vs chi')
#    plt.xlabel(r'$t$')
    
 #   print('tlate = '+str(tlate))
    
#    plt.figure()
#    plt.plot(sol_post.t,sol_post.y[11],label =  r'$\epsilon = $' + str(eps))#label = r'$\delta s_i = $ '+str(delta_sL))
#    plt.grid(b = True)
#    plt.title(r'Unstable potential, $\zeta_2$')#(r'$\zeta_2$ ')
   # plt.legend()
#    plt.xlabel(r'$t$')
    fnl = []
    for i in range(len(sol_post.t)):
        if sol_post.y[10][i] == 0:
            fnl.append(0)
        else:
            fnl.append(5./3*sol_post.y[11][i]/sol_post.y[10][i]**2)
#    plt.figure()
#    plt.plot(sol_post.t,fnl, label = 'Using '+str(integtype))
#    plt.grid(b = True)
#    plt.title(r'$f_{NL}$')
 #   plt.figure()
#    plt.plot(sol_post.t,sol_post.y[12])
#    plt.grid(b = True)
#    plt.title(r'$\zeta_3, q = $'+str(q) + ', Vrep cutoff = '+str(cutoff))
#    plt.xlabel(r'$t$')

#    zdd_guess = []
#    for i in range(len(sol_post.t)-1):
#        zdd_guess.append(sol_post.y[10][i+1]/sol_post.t[i+1]**2)
#    plt.figure()
#    plt.plot(sol_post.t[1:],zdd_guess)
#    plt.title('Zdd_guess')

    return [fNL,gNL,lnAs,tau_f]#,sympytime,resttime]

# --------
#sympytime =  np.zeros([len(input_eps),len(input_q),len(input_logV),len(input_thr),len(input_logvr)])
#resttime = np.zeros([len(input_eps),len(input_q),len(input_logV),len(input_thr),len(input_logvr)])
#output_fNL =  np.zeros([len(input_eps),len(input_q),len(input_logV),len(input_thr),len(input_logvr)])
#output_gNL =  np.zeros([len(input_eps),len(input_q),len(input_logV),len(input_thr),len(input_logvr)])
#output_logAs =  np.zeros([len(input_eps),len(input_q),len(input_logV),len(input_thr),len(input_logvr)])
#from Integ_onefunc import finalresult
#for i in range(len(input_eps)):
#    print(i)
#    for j in range(len(input_q)):
#        for k in range(len(input_logV)):
#            for l in range(len(input_thr)):
#                for s in range(len(input_logvr)):
#                    start = time.time()
#                    fNL_list,gNL_list,logAslist,symlist,restlist = finalresult(input_eps[i],input_q[j],input_logV[k],input_thr[l],input_logvr[s])
#                    print(input_eps[i],input_q[j],input_logV[k],input_thr[l],input_logvr[s])
#                   print('First took '+str(time.time() - start))
#                   start = time.time()

#                    sympytime[i][j][k][l][s] = symlist
#                    resttime[i][j][k][l][s] = restlist
#                    output_fNL[i][j][k][l][s] = fNL_list
#                    output_gNL[i][j][k][l][s] = gNL_list
#                    output_logAs[i][j][k][l][s] = logAslist
#                    print('then, '+str(time.time()-start))


#----------

#plt.figure()
#plt.plot(sol_post.t,ds,label = 'integration is '+str(integtype))
#plt.grid(b = True)
#plt.title(r'$\delta s_L$')
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,sol_post.y[10],label = 'integration is '+str(integtype))
#plt.grid(b = True)
#plt.title(r'$\zeta_L, q = $'+str(q) + ', Vrep cutoff = '+str(cutoff))
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.y[0],sol_post.y[2])
#plt.grid(b = True)
#plt.title('phi vs chi for eps = '+str(eps))
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,thd)
#plt.grid(b = True)
#plt.title('thd')#' for eps = '+str(eps))
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,sigmadot)
#plt.grid(b = True)
#plt.title('sigdot')#' for eps = '+str(eps))
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,hubble)
#plt.grid(b = True)
#plt.title('H')#' for eps = '+str(eps))
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,sol_post.y[1],label = 'phidot')
#plt.plot(sol_post.t,sol_post.y[3],label = 'chidot')
#plt.grid(b = True)
#plt.legend()
#plt.title('eps = '+str(eps))
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,vphi(sol_post.y[0],sol_post.y[2]),label = 'vphi')
#plt.plot(sol_post.t,vchi(sol_post.y[0],sol_post.y[2]),label = 'vchi')
#plt.grid(b = True)
#plt.title('eps = '+str(eps))
#plt.legend()
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,sol_post.y[0],label = 'phi')
#plt.plot(sol_post.t,sol_post.y[2],label = 'chi')
#plt.grid(b = True)
#plt.title('eps = '+str(eps))
#plt.legend()
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,pdd,label = 'eps = '+str(eps))
#plt.grid(b = True)
#plt.title('phidotdot')
#plt.legend()
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,denom,label = 'eps = '+str(eps))
#plt.grid(b = True)
#plt.title('denom')
#plt.legend()
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,first,label = 'eps = '+str(eps))
#plt.grid(b = True)
#plt.title('first')
#plt.legend()
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,second,label = 'eps = '+str(eps))
#plt.grid(b = True)
#plt.title('second')
#plt.legend()
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,third,label = 'eps = '+str(eps))
#plt.grid(b = True)
#plt.title('third')
#plt.legend()
#plt.xlabel(r'$t$')

#plt.figure()
#plt.plot(sol_post.t,fourth,label = 'eps = '+str(eps))
#plt.grid(b = True)
#plt.title('fourth')
#plt.legend()
#plt.xlabel(r'$t$')




#plt.figure()
#plt.plot(sol_post.t,sol_post.y[11],label = 'integration is '+str(integtype))
#plt.grid(b = True)
#plt.title(r'$\zeta_{NL}$')
#plt.xlabel(r'$t$')
#plt.show()
#plt.figure()
#plt.plot(sol_post.t,sol_post.y[12],label = 'integration is '+str(integtype))
#plt.grid(b = True)
#plt.title(r'$\zeta_{3}$')
#plt.xlabel(r'$t$')
#plt.show()


#Contour plot of repulsive potential
        
#x = np.linspace(-7, 0, 50)
#y = np.linspace(-5, 3, 30)
#X, Y = np.meshgrid(x, y)
#Vrep2 = np.vectorize(vrep)#Vrep)
#Z = Vrep2(X, Y)
#plt.figure()
#plt.contourf(X, Y, Z, 100,cmap='RdGy',extend='both') #Can replace 20 with np.linspace(Z.min(), Z.max(), 100) to avoid white space OR better just to add extend='both'. Also cmap='jet' might look better.
#cbar = plt.colorbar()
#cbar.set_label(r'$V_{rep}$', rotation=360)
#Let's also do background
#plt.plot(sol_post.y[0],sol_post.y[2])
#plt.arrow(3, 0, 1.0, 0, length_includes_head=True,
#          head_width=0.1, head_length=0.1)
#plt.xlabel(r'$\phi$')
#plt.ylabel(r'$\chi$')
#plt.title("Background and shifted repulsive potential, thr = "+str(th_rep))
#plt.show()