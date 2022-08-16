#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:55:21 2021

@author: joe
"""

from matplotlib import pyplot as plt  
from math import * 
import numpy as np 
from scipy import integrate
import sympy as sym 
import time
import sys 
from sympy import besseli
import scipy.integrate as integ
from scipy.integrate import solve_ivp

#Looking at https://arxiv.org/pdf/1510.03439.pdf
#Conversion happens before bounce

################################################

#f1 =vphi; f2 = vphiphi; f3 = vphiphiphi;f4 = vchi;f5 = vchichi
#f6 = vchichichi; f7 = vphichi; f8=vphichichi;f9=vphiphichi;f10=v;fr = vrep
#f11 = omega;f12 = omegaphi;f13 = omegaphiphi; f14 = omegaphiphiphi
#f15 = vphiphiphiphi ; f16 = vphiphiphichi ; f17 = vphiphichichi ; f18 = vphichichichi ; f19 =vchichichichi 
#^ Doing this should make it faster
def evolve(t,y):
    #print(t)
    phi,phidot,chi,chidot,delta_s,delta_sdot,delta_s2,delta_s2dot,delta_s3,delta_s3dot,zeta,zeta2,zeta3,N = y  
    
    vp = vphi(phi,chi);vpp = vphiphi(phi,chi);vppp = vphiphiphi(phi,chi)
    vc = vchi(phi,chi);vcc = vchichi(phi,chi);vccc = vchichichi(phi,chi)
    vpc = vphichi(phi,chi);vpcc = vphichichi(phi,chi);vppc = vphiphichi(phi,chi);pot = v(phi,chi)+vrep(phi,chi)
    Omega = omega(phi); Omegaphi = omegaphi(phi);Omegaphiphi = omegaphiphi(phi);Omegaphiphiphi = omegaphiphiphi(phi)
    vpppp = vphiphiphiphi(phi,chi); vpppc = vphiphiphichi(phi,chi); vppcc = vphiphichichi(phi,chi); vpccc = vphichichichi(phi,chi); vcccc = vchichichichi(phi,chi)
    
    sigdot = sqrt(phidot**2+Omega**2*chidot**2)   
    
    rho = 0.5*phidot**2+0.5*Omega**2*chidot**2+pot
    H = -sqrt(abs(rho)/3.)
    #a = exp(N)
    
    phidotdot = -3*H*phidot-vp+Omega*Omegaphi*chidot**2
    chidotdot=-(3*H+2*Omegaphi/Omega)*chidot-vc/Omega**2
                        #   ^Typo in .nb, should be a phidot?
    
    sigdotdot = phidot/sigdot*phidotdot + Omega*Omegaphi*phidot*chidot**2/sigdot + Omega**2*chidot*chidotdot/sigdot
    

        
    esp =-Omega*chidot/sigdot
    esc = phidot/sigdot/Omega
    esigp = phidot/sigdot
    esigc = chidot/sigdot
    Gpcc = G122(phi)
    Gpccp = G1221(phi)
    Gcpcpp = G21211(phi)
    Gpccpp = G12211(phi)
    Gcpcp = G2121(phi)
    Gcpc = G212(phi)
    Rpcpc = R1212(phi)
    Rpcpcp = R1212(phi)
    Rpccp = R12121(phi)
    Rcppc = R2112(phi)
    Rcpcp = R2121(phi)
   # factor = -3*omegaphi*omegaphiphi*vp-omegaphi**2*vpp-omega*omegaphiphiphi*vp-2*omega*omegaphiphi*vpp-omegaphi**2*vpp-omega*omegaphi*vppp
    
    ############
    #Vsig = (phidot*vp+chidot*vc)/sigdot 
   # if vrep(phi,chi) >= 2.4e-7:#.1*vr_first:
    #    print('STILL ON AT t = '+str(t))
  #      print('and vrep = '+str(vrep(phi,chi)))
  #      print('phi = '+str(phi))
  #      print('chi = '+str(chi))
  
    Vsig = (phidot*vp+chidot*vc)/sigdot 
    Vs = -chidot*vp*Omega/sigdot+phidot*vc/sigdot/Omega
    Vsdot = -chidotdot/sigdot*vp*Omega+chidot/sigdot**2*sigdotdot*Omega*vp-chidot*phidot/sigdot*Omegaphi*vp-chidot/sigdot*Omega*(vpp*phidot+vpc*chidot)+phidotdot/sigdot*vc/Omega-phidot*sigdotdot/sigdot**2*vc/Omega-phidot**2/sigdot*Omegaphi/Omega**2*vc+phidot/sigdot/Omega*(vcc*chidot+vpc*phidot)
    thetadotdot = -Vsdot/sigdot+Vs/sigdot**2*sigdotdot
    thetadot = -Vs/sigdot
    v_cov_ssig = esp*esigp*vpp + esc*esigc*vcc +(esp*esigc + esc*esigp)*vpc - esc*esigc*Gpcc*vp - Gcpc*vc*(esp*esigc +esc*esigp )
    v_cov_sigsig = esigp**2*vpp + esigc**2*vcc+2*esigp*esigc*vpc -2*esigp*esigc*Gcpc*vc-esigc**2*Gpcc*vp
    v_cov_sssig = esp**2*esigp*vppp+esc**2*esigc*(vccc-3*Gpcc*vpc+2*Gpcc*Gcpc*vc)+esc**2*esigp*(vpcc - Gpccp*vp-Gpcc*vpp-2*Gcpc*vcc+2*Gcpc*Gpcc*vp) + 2*esp*esc*esigc*(vpcc-2*Gcpc*vcc-Gpcc*vpp+Gcpc*Gpcc*vp)+esp**2*esigc*(vppc-2*Gcpc*vpc+2*Gcpc*Gcpc*vc)+2*esp*esc*esigp*(vppc-Gcpcp*vc-2*Gcpc*vpc+Gcpc*Gcpc*vc)#esp**2*esigp*vppp + esc**2*esigc*vccc+vpcc*(esc**2*esigp+2*esc*esp*esigc)+vppc*(esp**2*esigc+2*esc*esp*esigp) - esp*esc*esigc*(Gpccp*vp+vpp*Gpcc)-esc**2*esigc*Gpcc*vpc-esp**2*esigc*(Gcpcp*vc + Gcpc*vpc) - esp*esc*esigp*(Gcpcp*vc + Gcpc*vpc)-esc*esp*esigc*Gcpc*vcc - esc**2*esigp*Gcpc*vcc - 2*esc**2*esigp*Gpcc*vpp - 2*esc**2*esigc*Gpcc*vpc - 4*esp*esc*esigp*Gcpc*vpc - 4*esp*esc*esigc*Gcpc*vcc + 2*esc**2*esigc*Gpcc*Gcpc*vc +4*esp*esc*esigp*Gcpc**2*vc+4*esp*esc*esigc*Gcpc*Gpcc*vp
    v_cov_ss = (Omega*chidot/sigdot)**2*vpp - 2*phidot*chidot/sigdot**2*(vpc - Gcpc*vc)+(phidot/(sigdot*Omega))**2*(vcc -Gpcc*vp)
    v_cov_sss = esp**3*vppp + esc**3*(vccc-3*Gpcc*vpc+2*Gpcc*Gcpc*vc) + esp*esc**2*(3*vpcc-Gpccp*vp-3*Gpcc*vpp-Gcpc*(6*vcc-4*Gpcc*vp)) + esp**2*esc*(3*vppc-2*Gcpcp*vc -Gcpc*(6*vpc-4*Gcpc*vc))#esp**3*vppp + esc**3*(vccc- Gpcc*vpc - 6*Gpcc*(vpc - Gcpc*vc)) + esp**2*esc*(3*vppc  - 2*(vc*Gcpcp+Gcpc*vpc))+ esp*esc**2*(3*vpcc - vp*Gpccp-vpp*Gpcc)  - esc**2*esp*(2*Gcpc*vcc+ 2*Gpcc*vpp) 
    #big1 = -3*esc**2*esp**2*Gpcc*vppp - 6*esc**2*esp**2*Gpcc*vppc - 3*esc**4*Gpcc*vpcc - 6*esc*esp**3*Gcpc*vppc - 12*esc**2*esp**2*Gcpc*vpcc - 6*esc**3*esp*Gcpc*vccc + esc**4*Gpcc*Gpccp*vp + 2*esc**3*esp*Gpcc*Gcpcp*vc + 2*esc**3*esp*Gpcc*Gcpc*vpc + 2*esp*esc**3*Gcpc*Gpcc*vpc + 4*esc**2*esp**2*Gcpc**2*vcc + esc**4*Gpcc**2*vpp+4*esc**3*esp*Gpcc*Gcpc*vpc + 8*esc**2*esp**2*Gcpc*Gpcc*vpp+8*esc*esp**3*Gcpc**2*vpc+8*esp*esc**3*Gcpc*Gpcc*vpc+8*esp**2*esc**2*Gcpc**2*vcc+4*esc**4*Gpcc*Gcpc*vcc - 8*esp*esc**3*Gcpc*Gpcc*Gcpc*vc - 8*esp**2*esc**2*Gcpc*Gpcc*Gcpc*vp - 4*esc**4*Gpcc**2*Gcpc*vp - 4*esc**3*esp*Gpcc*Gcpc**2*vc - 8*esp**3*esc*Gcpc**3*vc
    #big2 = 4*esp**2*esc**2*Gcpc*Gpccp*vp + 2*esc**3*esp*Gpcc*Gcpcp*vc + 4*esp**3*esc*Gcpc*Gcpcp*vc + 4*esp*esc**3*Gcpc*Gpcc*vpc + 4*esp**2*esc**2*Gcpc**2*vcc+4*esp**2*esc**2*Gcpc*Gpcc*vpp + 2*esc**3*esp*Gpcc*Gcpc*vpc + 2*esc**4*Gpcc*Gcpc*vcc+4*esc*esp**3*Gcpc**2*vpc
    #big3 = 2*esc**4*Gpcc**2*vpp + 4*esc**3*esp*Gcpc*Gpcc*vpc + 4*esp*esc**3*Gpcc*Gcpc*vpc + 8*esc**2*esp**2*Gcpc**2*vcc - 8*esp**2*esc**2*Gcpc**2*Gpcc*vp - 8*esc**3*esp*Gpcc*Gcpc**2*vc
    v_cov_ssss = esp**4*vpppp+esc**4*(vcccc-6*Gpcc*vpcc+5*Gpcc*Gcpc*vcc+3*Gpcc*Gpcc*vpp-4*Gpcc*Gpcc*Gcpc*vp+Gpcc*Gpccp*vp)+esp*esc**3*(4*vpccc-12*Gcpc*vccc-12*Gpcc*vppc-4*Gpccp*vpc+2*Gcpc*Gpccp*vc+36*Gcpc*Gpcc*vpc+6*Gpcc*Gcpcp*vc-20*Gcpc*Gcpc*Gpcc*vc)+esp**3*esc*(4*vpppc-2*Gcpcpp*vc-12*Gcpc*vppc-8*Gcpcp*vpc+12*Gcpc*Gcpcp*vc+16*Gcpc*Gcpc*vpc-8*Gcpc*Gcpc*Gcpc*vc) + esp**2*esc**2*(6*vppcc-Gpccpp*vp-6*Gpcc*vppp-22*Gcpc*vpcc-3*Gpccp*vpp-8*Gcpcp*vcc+4*Gpcc*Gcpcp*vp+8*Gcpc*Gpccp*vp+16*Gcpc*Gpcc*vpp+28*Gcpc*Gcpc*vcc-16*Gcpc*Gcpc*Gpcc*vp)#esp**4*vpppp + 4*esc*esp**3*vpppc + 6*esc**2*esp**2*vppcc+4*esc**3*esp*vpccc+esc**4*vcccc - esp**2*esc**2*factor - esc**4*Gpcc*vpcc + 2*esp*esc**3*(omegaphi**2*vpc+omega*omegaphiphi*vpc+omega*omegaphi*vppc) - 2*esp**3*esc*( vc*(omegaphiphiphi/omega - omegaphiphi*(omegaphi/omega)**2-2*omegaphi/omega*(omegaphiphi/omega-(omegaphi/omega)**2)) + 2*vpc*(omegaphiphi/omega - (omegaphi/omega)**2)+omegaphi/omega*vppc)-2*esc**3*esp*Gcpc*vccc - 4*esp**2*esc**2*((omegaphiphi/omega - (omegaphi/omega)**2)*vcc+omegaphi/omega*vpcc)-2*esp*esc**3*(-omegaphi**2 - omega*omegaphiphi)*vpc - 2*esc**2*esp**2*(-omegaphi**2 - omega*omegaphiphi)*vpp - 4*esp**3*esc*(omegaphiphi/omega - (omegaphi/omega)**2)*vpc - 4*esp**3*esc*(omegaphiphi/omega - (omegaphi/omega)**2)*vcc + esp*esc**3*(omegaphiphi/omega - (omegaphi/omega)**2)*omegaphi/omega*vc -2*esp**2*esc**2*Gpcc*vppp - 4*esp*esc**3*Gpcc*vppc - 2*esc**4*Gpcc*vpcc - 4*esc*esp**3*Gcpc*vppc - 8*esp**2*esc**2*Gcpc*vpcc - 4*esc**3*esp*Gcpc*vccc + 4*esp**2*esc**2*Gcpc*Gpccp*vp + 4*esp**3*esc*Gcpc*Gcpcp*vc + 2*esp*esc**3*Gpcc*Gcpcp*vc + 2*esp*esc**3*Gpcc*Gcpc*vpc+4*esc**3*esp*Gcpc*Gpcc*vpc + 4*esp**3*esc*Gcpc**2*vpc + 4*esp**2*esc**2*Gcpc*Gpcc*vpp + 4*esc**2*esp**2*Gcpc**2*vcc + 2*esc**4*Gpcc*Gcpc*vcc + big1 + big2 + big3
    eeeeeDR = esigp*(Rpcpcp - 2*Gcpc*Rpcpc)*(esp**2*esigc**2 + esc**2*esigp**2 - 2*esp*esc*esigp*esigc)#(omegaphi*omegaphiphi + omega*omegaphiphiphi)*(2*esp*esp*esc*esigc*esigp - 4*esp*esp*esigc*esigc*esp -4*esc*esc*esigp*esigp*esp) -esc*esp*esigc*esigc*esc*Gpcc*Rpcpc - esc*esc*esigc*esigp*esc*Gpcc*Rpccp  - 2*esc*esp*esigp*esigc*esp*Gcpc*Rcppc - 2*esc*esc*esigp*esigp*esp*Gcpc*Rcpcp - 2*esp*esp*esigp*esigc*esc*Gcpc*Rpcpc - 2*esp*esp*esigc*esigc*esp*Gcpc*Rpcpc - 2*esp*esc*esigp*esigp*esc*Gcpc*Rpccp- 2*esp*esc*esigc*esigp*esp*Gcpc*Rpccp - 4*esc*esp*esigc*esigc*esc*Gpcc*Rcppc - 4*esc*esc*esigc*esigp*esc*Gpcc*Rcpcp - 2*esp*esc*esigc*esigc*esc*Gpcc*Rpcpc- 2*esp*esp*esigc*esigp*esc*Gcpc*Rpccp - 2*esc*esc*esigp*esigc*esc*Gpcc*Rcppc - 2*esc*esp*esigp*esigp*esc*Gcpc*Rcpcp   
    eeeeR = Rpcpc*(esp**2*esigc**2 + esc**2*esigp**2 - 2*esp*esc*esigp*esigc)#-omega*omegaphiphi*chidot**2/sigdot**4*(omega*chidot**2+2*phidot**2)
    eeDDR = 0.
    eeRR1 = Omegaphiphi**2*(esigp*esc - esigc*esp)**2*esp**2
    eeRR2 = (Omegaphiphi**2 +Omega**2*Omegaphi*Omegaphiphi)*esp**2*(esp*esigc - esc*esigp)**2

    this = v_cov_ss + 3*thetadot**2+sigdot**2*eeeeR
    term1 = v_cov_sss - 10/sigdot*v_cov_ss*thetadot - 18/sigdot*thetadot**3 - eeeeR*2*sigdot*thetadot + sigdot**2*eeeeeDR
    term2 = 2./3*v_cov_sigsig +2*Vsig**2/sigdot**2+H*Vsig/sigdot-v_cov_ss-8./3*thetadot**2-sigdot**2*eeeeR
    term3 = -22./(3*sigdot**2)*thetadot*thetadotdot-7./(6*sigdot)*v_cov_sssig - 11./(3*sigdot**3)*v_cov_ss*Vsig - 13./(3*sigdot**3)*Vsig*thetadot**2 - H*v_cov_ss/sigdot**2+18./sigdot**2*H*thetadot**2-4*Vsig/(3*sigdot)*eeeeR - sigdot/6*eeeeeDR
    term4 = v_cov_ssss/6 - 7/(3*sigdot)*v_cov_sss*thetadot + 5/(3*sigdot**2)*v_cov_ss**2 + 19/sigdot**2*v_cov_ss*thetadot**2+24/sigdot**2*thetadot**4+1./3*eeeeR*(v_cov_ss+thetadot**2)-2./3*sigdot*thetadot*eeeeeDR + sigdot**2*(.5*eeDDR - eeRR1 + eeRR2)



    zetadot = -2.*H/sigdot*thetadot*delta_s
    zeta2dot = 2.*H/sigdot**2*(-sigdot*thetadot*delta_s2 - Vsig/(2*sigdot)*delta_s*delta_sdot+(.5*v_cov_ss+2*thetadot**2)*delta_s**2)
    zeta3dot = 2.*H/sigdot**2*(-sigdot*thetadot*delta_s3 - Vsig/(2*sigdot)*(delta_sdot*delta_s2 + delta_s*delta_s2dot)+(v_cov_ss +4*thetadot**2)*delta_s*delta_s2 + thetadot/(6*sigdot)*delta_s*delta_sdot**2 + (11./6*thetadot*Vsig/sigdot**2 - v_cov_ssig/(2*sigdot))*delta_s**2*delta_sdot + (v_cov_sss/6-2*thetadot*v_cov_ss/sigdot - 4*thetadot**3/sigdot)*delta_s**3)
    

    delta_sdotdot=-3*H*delta_sdot-(this)*delta_s
    source2 = thetadot/sigdot*delta_sdot**2 + 2/sigdot*(thetadotdot + thetadot/sigdot*Vsig - 1.5*H*thetadot)*delta_s*delta_sdot + (.5*v_cov_sss - 5*thetadot/sigdot*v_cov_ss-9*thetadot**3/sigdot - (esp**2*esigc**2+esc**2*esigp**2-2*esp*esc*esigp*esigc)*(sigdot*thetadot*Rpcpc - .5*sigdot**2*esp*(Rpcpcp-2*Gcpc*Rpcpc)))*delta_s**2
    delta_s2dotdot=-3*H*delta_s2dot-delta_s2*(v_cov_ss+3*thetadot**2 + (sigdot**2*(esp**2*esigc**2+esc**2*esigp**2 - 2*esp*esc*esigp*esigc)*Rpcpc)) - source2
    delta_s3dotdot = -3*H*delta_s3dot - this*delta_s3 - 2*thetadot/sigdot*delta_sdot*delta_s2dot - (2*thetadotdot/sigdot+2/sigdot**2*Vsig*thetadot-3*H*thetadot/sigdot)*(delta_sdot*delta_s2 + delta_s*delta_s2dot)-term1*delta_s*delta_s2 - Vsig/(3*sigdot**3)*delta_sdot**3 -1/sigdot**2*term2*delta_s*delta_sdot**2-term3*delta_s**2*delta_sdot-term4*delta_s**3
   
    Ndot = H
    return phidot,phidotdot,chidot,chidotdot,delta_sdot,delta_sdotdot,delta_s2dot,delta_s2dotdot,delta_s3dot,delta_s3dotdot,zetadot,zeta2dot,zeta3dot,Ndot#,thdot #dydt


#aH = []
#negaH = []
#H = []

#a = []

#for i in range(len(result.t)):
#    r = 0.5*result.y[1][i]**2+0.5*omega(result.y[0][i])**2*result.y[3][i]**2+v(result.y[0][i],result.y[2][i])+vrep(result.y[0][i],result.y[2][i])
#    h = -sqrt(abs(r)/3)
#    #print(result.y[-1][i])
#    H.append(h)
#    a.append(exp(result.y[-1][i]))
#    aH.append(h*exp(result.y[-1][i]))
#    negaH.append(-h*exp(result.y[-1][i]))

#kin = [];pot = [];EOS = [];vr = []
#for i in range(len(result.t)):
#    kin.append(0.5*result.y[1][i]**2)
#    pot.append(v(result.y[0][i],result.y[2][i]))  
#    vr.append(vrep(result.y[0][i],result.y[2][i]))
#    w = (kin[i] - pot[i])/(kin[i] + pot[i])
#    EOS.append(1.5*(1+w))

 #check when aH[0]*exp(1) happens            
    
#print('phi at tend = '+str(result.y[0][-1]))        
#plt.figure()
#plt.plot(result.t,np.log(negaH))
#plt.title('N')
#plt.show()
#--------------------------------------

#from scipy import integrate as integ
#convint=2700
#kinint=4250
#tconv = result.t[0:convint]
#tkin = result.t[0:kinint]
#numerator = integ.cumtrapz(thetadot[0:convint],tconv)[-1]
#denominator = integ.cumtrapz(thetadot[0:kinint],tkin)[-1]
#efficiency = numerator/denominator

#aHend = H[convint]*a[convint]
#aHbeg = H[0]*a[0]
#N = log(aHend/aHbeg)
#print('N = '+str(N))
#print('Taking t_endkin = '+str(result.t[kinint]))
#print('Taking t_endconv = '+str(result.t[convint]))

##############
#index = np.unravel_index(np.argmax(thetadot, axis=None), np.array(thetadot).shape)
#indexneg = np.where(np.array(thetadot) < 0.)[0][0]
#fulltheta = integ.cumtrapz(thetadot[0:indexneg],result.t[0:indexneg])[-1]
#print(.9*fulltheta)
#conv = 1000
#m = min(result.t[index[0] +conv], result.t[indexneg])
#mint = np.where(result.t == m)[0][0]
#rhs = integ.cumtrapz(thetadot[index[0] -conv :mint],result.t[index[0] -conv :mint])[-1]
#print(rhs)
#print('If the two above are not close, chage conv')
#aHend = H[index[0] +conv]*a[index[0] +conv]
#aHbeg = H[index[0] -conv]*a[index[0] -conv]
#N = log(aHend/aHbeg)
#print('N = '+str(N))
#print('tconvbeg = '+str(result.t[index[0] -conv]))
#print('tconvend = '+str(m))
#print('tkinvbeg = '+str(result.t[0]))
#print('tkinvend = '+str(result.t[indexneg]))
#plt.figure()
#plt.plot(result.t,thetadot)
#plt.title('thdot')
#################



#Plotefficienicy
#Maybe integrate exactly?
#
#x = sympy.Symbol('x')
#sympy.integrate(function(x),x)
#returns a func of x
#numerator = integrate.cumtrapz(t2[:832], t[:832])
#denominator = integrate.cumtrapz(t2,t)
#efficiency = numerator[-1]/denominator[-1]
#Should be 0.9

#OR integrate(cos(x),(x,lowerbound,upperbound))


#conv starts when chi changes
#convstart

#numerator = integrate.cumtrapz(t2[convstart:convend], t[convstart:convend])
#denominator = integrate.cumtrapz(t2,t)
#efficiency = numerator[-1]/denominator[-1]
#Should be 0.9


#Better idea? -> 0.9 is a requirement which determines the start and end time of the
#conversion, and we can THEN check if this corresponds to one efold.
#If we do net get N = 1, then modify v. But how much does initial delta s change this??

#denominator = integrate.cumtrapz(t2,t)
#For this loop, let's say convend is approx t=0, then find the real tend later.
#for j in range(len(t)):
#    numerator = integrate.cumtrapz(t2[j:], t[j:])
#    if numerator[-1]/denominator[-1]<= 0.9:
#        print (j)
#        break


#interval = (convstart/convend)**(2/3.) #should be about e, if indeed N_conv is 1. 


#Contour plot of repulsive potential
        
#x = np.linspace(-2, 2, 50)
#y = np.linspace(-2, 2, 30)

#X, Y = np.meshgrid(x, y)
#Vrep2 = np.vectorize(vrep)#Vrep)
#Z = Vrep2(X, Y)
#plt.contourf(X, Y, Z, 20,cmap='RdGy',extend='both') #Can replace 20 with np.linspace(Z.min(), Z.max(), 100) to avoid white space OR better just to add extend='both'. Also cmap='jet' might look better.
#cbar = plt.colorbar()
#cbar.set_label(r'$V_{rep}$', rotation=360)
#Let's also do background
#plt.plot(phi,chi)
#plt.arrow(3, 0, 1.0, 0, length_includes_head=True,
#          head_width=0.1, head_length=0.1)
#plt.xlabel(r'$\phi$')
#plt.ylabel(r'$\chi$')
#plt.title("Background and shifted repulsive potential")
#plt.show()



def output_bessel(b,d,c,lnv, r,V1):
    global R1212,R12121,R1221,R2112,R2121  ,G122,G212,G2121,G1221 ,G12211,G21211 ,omega,omegaphi,omegaphiphi,omegaphiphiphi,vrep,v,vphi,vphiphi,vphiphiphi,vphiphiphiphi,vchi,vchichi,vchichichi,vchichichichi,vphichi,vphichichi,vphiphichi,vphiphiphichi,vphiphichichi,vphichichichi,kappa3,rep_max,t_endrep

    # Global variables.

    r_tol = 1e-8#6
    a_tol = 1e-8#6
    integtype = 'RK45' #For non stiff, use ‘RK23’, ‘RK45’, or ‘DOP853’


    #eps=36.#17.22222222222222 #float(sys.argv[1])# 5.#float(sys.argv[1])#8.0
    #^fNL seems fully indep of this
    #ACTUALLY eps IS NOT EVEN A PARAM IN THIS MODEL?
    #Vo = 0.#exp(-20)#exp(float(sys.argv[2]))#1e-9#round(float(sys.argv[2])*1e-6,10)#(eps-3)/eps**2*exp(1.6*sqrt(eps/2))/304**2#2e-9
    
   # d = b
   # c=1
    tend = -1000;tbang = -1;
    
    repulsive_amplitude = exp(lnv)#exp(float(sys.argv[4]))#206e-11#exp(-22.0)#exp((float(sys.argv[2])))#200000*exp(-33.6)#exp(float(sys.argv[2]))#exp(-31.5)#exp(float(sys.argv[3]))#1e-9#round(float(sys.argv[3])*1e-6,10)#2e-11#5e-13#1e-4*Vo#5e-7#5e-8


    #################################################
    # Functions

    phi, chi = sym.symbols('phi chi')
    #v = -Vo*sym.exp(-sym.sqrt(2*eps)*phi)#sym.Piecewise((-Vo*sym.exp(-sym.sqrt(2.*epsilon)*phi),phi>= 1.3),(0.,True))
    v = 0
    vphi = sym.diff(v,phi)
    vphiphi = sym.diff(v,phi,2)
    vphiphiphi = sym.diff(v,phi,3)
    vphiphiphiphi = sym.diff(v,phi,4)


    omega = 1 - b*besseli(0, d*sym.exp(c*phi/2))
    #omega = 1 - b*sym.exp(d/2*phi)
    omegaphi = sym.diff(omega,phi)
    omegaphiphi = sym.diff(omega,phi,phi)
    omegaphiphiphi = sym.diff(omega,phi,phi,phi)
    
    G212 = omegaphi/omega
    G122 = -omega*omegaphi
    G2121 = sym.diff(G212,phi)
    G1221 = sym.diff(G122,phi)
    G21211 = sym.diff(G212,phi,phi)
    G12211 = sym.diff(G122,phi,phi)
    
    R1212 = -omega*omegaphiphi
    R12121 = -omegaphi*omegaphiphi-omega*omegaphiphiphi
    R1221 = -R1212
    R2112 = R1221
    R2121 = -R2112 
    if V1 == True:
        x = -phi/2+sqrt(3.)/2*chi
        #           ^THIS SIGN IS WHAT IS ACTUALLY USED
        vrep = repulsive_amplitude*(1/x**2+r/x**6)
    else:
    
        x = sym.sinh(phi/2+sqrt(3.)/2*chi)
        vrep = repulsive_amplitude*(1/x**2+r/x**4)
    
    #vrep = repulsive_amplitude*(1/(sym.sinh(x))**2+r/(sym.sinh(x))**4)

    vrepphi = sym.diff(vrep,phi)  #The first argument of diff can be a function I defined, just be sure to include its args, eg Vrep(phi,chi).
    vrepchi = sym.diff(vrep,chi)
    vrepphiphi = sym.diff(vrep,phi,phi)
    vrepphichi = sym.diff(vrep,phi,chi)
    vrepchichi = sym.diff(vrep,chi,chi)
    vrepphiphiphi = sym.diff(vrep,phi,phi,phi)
    vrepphiphichi = sym.diff(vrep,phi,phi,chi)
    vrepphichichi = sym.diff(vrep,phi,chi,chi)
    vrepchichichi = sym.diff(vrep,chi,chi,chi)

    vrepphiphiphiphi = sym.diff(vrep,phi,phi,phi,phi)
    vrepphiphiphichi = sym.diff(vrep,phi,phi,phi,chi)
    vrepphiphichichi = sym.diff(vrep,phi,phi,chi,chi)
    vrepphichichichi = sym.diff(vrep,phi,chi,chi,chi)
    vrepchichichichi = sym.diff(vrep,chi,chi,chi,chi)


    vtotphi = vphi+vrepphi
    vtotphiphi = vphiphi+vrepphiphi
    vtotphiphiphi = vphiphiphi+vrepphiphiphi
    vtotphiphiphiphi = vphiphiphiphi+vrepphiphiphiphi
    #Then make them callable functions; note they are arrays, may have to make intofloats later
    v = sym.lambdify((phi,chi),v)
    vrep = sym.lambdify((phi,chi),vrep)
    vphi = sym.lambdify((phi,chi),vtotphi)
    vchi = sym.lambdify((phi,chi),vrepchi)
    vphiphi = sym.lambdify((phi,chi),vtotphiphi)
    vphichi = sym.lambdify((phi,chi),vrepphichi)
    vchichi = sym.lambdify((phi,chi),vrepchichi) 
    vphiphiphi = sym.lambdify((phi,chi),vtotphiphiphi)
    vphiphichi = sym.lambdify((phi,chi),vrepphiphichi)
    vphichichi = sym.lambdify((phi,chi),vrepphichichi)
    vchichichi = sym.lambdify((phi,chi),vrepchichichi)

    vphiphiphiphi = sym.lambdify((phi,chi),vtotphiphiphiphi)
    vphiphiphichi = sym.lambdify((phi,chi),vrepphiphiphichi)
    vphiphichichi = sym.lambdify((phi,chi),vrepphiphichichi)
    vphichichichi = sym.lambdify((phi,chi),vrepphichichichi)
    vchichichichi = sym.lambdify((phi,chi),vrepchichichichi)

    omega = sym.lambdify((phi),omega)
    omegaphi = sym.lambdify((phi),omegaphi)
    omegaphiphi = sym.lambdify((phi),omegaphiphi)
    omegaphiphiphi = sym.lambdify((phi),omegaphiphiphi)

    G212 = sym.lambdify((phi),G212)
    G122 = sym.lambdify((phi),G122)
    G2121 = sym.lambdify((phi),G2121)
    G1221 = sym.lambdify((phi),G1221)
    G21211 = sym.lambdify((phi),G21211)
    G12211 = sym.lambdify((phi),G12211)
    
    R1212 = sym.lambdify((phi),R1212)
    R12121 = sym.lambdify((phi),R12121)
    R1221 = sym.lambdify((phi),R1221)
    R2112 = sym.lambdify((phi),R2112)
    R2121 =sym.lambdify((phi),R2121)


    #################################################

    # Dynamic fields (including time)
    
    #tend = -46. #Note, I think when text says it goes to t=-46, I think it is only referring to the curvature plot, not the field space one
    

    phistart = sqrt(3)#.797828#sqrt(2/eps)*log(-tstart*sqrt(Vo*eps**2/(eps-3)))
    phidotstart = sqrt(2/3)/tend#sqrt(2/eps)/tstart# 1.175*sqrt(2/eps)/tstart
    

    delta_sstart =1#1e-10 #From 1e-5 to 1e-10, no difference to fNL
    delta_sdotstart = -delta_sstart/tend
    delta_s2start = 0.#delta_sstart**2*kappa3*sqrt(eps)/8
    delta_sdot2start = 0.
    delta_s3start = 0.
    delta_sdot3start = 0.

    #vr_first = vrep(phistart,0.)

    ##################################################
    first = np.append(phistart, phidotstart);second =np.append(0.,0.);
    third = np.append(delta_sstart,delta_sdotstart)
    fourth = np.append(delta_s2start,0)
    fifth = np.append(delta_s3start,0.)
    sixth = np.append(0.,0.)
    this = np.append(first,second)
    this2 = np.append(this,third)
    this3 = np.append(this2,fourth)
    this4 = np.append(this3,fifth)
    this5 = np.append(this4,sixth)
    logainit = 0#1.#???
    seventh = np.append(0.,logainit)
    y0 = np.append(this5,seventh)

    t = np.linspace(tend,tbang, 10000) #Last number doesn't affect integration, just how many points are shown in plot
    result = solve_ivp(evolve, [t[0], t[-1]], y0, t_eval=t,rtol = r_tol,atol = a_tol,method = integtype)
    
    term1 = [] ; term2 = [] ; term3 = []
    zeta2dot = []
    t1plust3 = []; thdarr = []
    Nconv = []
    aarr = [] ; Harr = []
    vparr = [] ; vcarr = []
    omarr = []
    argument = []
    ricci = []
    adot = []

  #  for i in range(len(result.t)):
  #      phidot = result.y[1][i]
  #      chidot = result.y[3][i]
  #      phi = result.y[0][i]
  #      chi = result.y[2][i]
  #      Omega = omega(phi)
  #      sigdot = sqrt(phidot**2+Omega**2*chidot**2)
  #      pot = v(phi,chi)
    
  #      rho = 0.5*phidot**2+0.5*Omega**2*chidot**2+pot
  #      H = -sqrt(abs(rho)/3.)
  #      a = exp(result.y[-1][i])
  #      aarr.append(a)
  #      adot.append(H*a)
  #      Harr.append(-H)
  #      Nconv.append(log(abs(H*a)))
  #      vparr.append(vphi(phi,chi))
  #      vcarr.append(vchi(phi,chi))
  #      omarr.append(Omega)
  #      ompp = omegaphiphi(phi)
  #      ricci.append(-2*ompp/omarr[i])
  #      argument.append(phi/2 + sqrt(3)/2*chi)
  #      Vsig = (phidot*vphi(phi,chi)+chidot*vchi(phi,chi))/sigdot 
  #      Vs = -chidot*vphi(phi,chi)*Omega/sigdot+phidot*vchi(phi,chi)/sigdot/Omega
  #      v_ss = (Omega*chidot/sigdot)**2*vphiphi(phi,chi) - 2*phidot*chidot/sigdot**2*(vphichi(phi,chi) - G212(phi)*vchi(phi,chi))+(phidot/(sigdot*Omega))**2*(vchichi(phi,chi) -G122(phi)*vphi(phi,chi))
  #      thdot = -Vs/sigdot
  #      thdarr.append(thdot)
  #      term1.append(-2*H/sigdot*thdot*result.y[6][i])
  #      term2.append(-H/sigdot**3*Vsig*result.y[4][i]*result.y[5][i])
  #      term3.append(2*H/sigdot**2 * (.5*v_ss + 2*thdot**2)*result.y[4][i]**2)
  #      t1plust3.append(term1[i] + term3[i])
  #      zeta2dot.append(term1[i] + term2[i] + term3[i])
 #   plt.figure()
 #   plt.plot(result.t,Nconv)
 #   plt.title('N')
 #   plt.figure()
 #   plt.plot(result.t,aarr)
 #   plt.title('a')
 #   plt.figure()
 #   plt.plot(result.t,adot)
 #   plt.title('adot')
 #   plt.figure()
 #   plt.plot(result.t,Harr)
 #   plt.title('|H|')
   # plt.figure()
   # plt.plot(result.t,result.y[1])
   # plt.title('phidot')
 #   plt.figure()
 #   plt.plot(result.t,ricci)
 #   plt.title('Ricci')
   # plt.figure()
   # plt.plot(result.y[0],ricci)
   # plt.title('Ricci vs phi')
   # plt.figure()
   # plt.plot(result.t,result.y[3])
   # plt.title('chidot')
    #plt.figure()
    #plt.plot(result.t,vparr)
    #plt.title('Vphi')
    #plt.figure()
    #plt.plot(result.t,vcarr)
    #plt.title('Vchi')
   # plt.figure()
   # plt.plot(result.t,omarr)
   # plt.title('Omega')
 #   plt.figure()
 #   plt.plot(result.y[0],omarr)
 #   plt.title('Omega vs phi')
   # plt.figure()
   # plt.plot(result.t,term1,label = 'lnv = '+str(lnv))
   # plt.title('Term 1')
   # plt.figure()
   # plt.plot(result.t,result.y[0])
   # plt.title('phi')
 #   plt.figure()
 #   plt.plot(result.t,result.y[2])
 #   plt.title('chi')
 #   plt.figure()
 #   plt.plot(result.t,argument)
 #   plt.title('Argument in sinh')
 #   plt.figure()
 #   plt.plot(result.y[0],result.y[2])
 #   plt.title('phi vs chi')
 #   plt.figure()
 #   plt.plot(result.t,thdarr,label = 'lnv = '+str(lnv))
 #   plt.title('theta dot')
  #  plt.figure()
  #  plt.plot(result.t,t1plust3,label = 'lnv = '+str(lnv))
  #  plt.title('Term 1 plus Term3')
    
  #  plt.figure()
  #  plt.plot(result.t,term2,label = 'lnv = '+str(lnv))
  #  plt.title('Term 2')
    
  #  plt.figure()
  #  plt.plot(result.t,term3,label = 'lnv = '+str(lnv))
  #  plt.title('Term 3')
    
 #   plt.figure()
 #   plt.plot(result.t,zeta2dot,label = 'lnv = '+str(lnv))
  #  plt.title('zeta2 dot')
    
 #   plt.figure()
 #   plt.plot(result.t,result.y[11],label = 'lnv = '+str(lnv))
 #   plt.title('zeta2, lnv = '+str(lnv))
  
 #   plt.figure()
 #   plt.plot(result.t,result.y[10],label = 'lnv = '+str(lnv))
 #   plt.title('zetaL')
  
 #   plt.figure()
 #   plt.plot(result.t,result.y[6],label = 'lnv = '+str(lnv))
 #   plt.title('delta s (2)')
 #   plt.figure()
 #   plt.plot(result.t,result.y[8],label = 'lnv = '+str(lnv))
 #   plt.title('delta s (3)')
 #   plt.figure()
 #   plt.plot(result.t,result.y[4],label = 'lnv = '+str(lnv))
 #   plt.title('delta s (1)')
    
    
#    denominator = integ.cumtrapz(thdarr,result.t)[-1]#t)[-1]
    
#    tnum = np.linspace(-165,-97)
 #   tnum = np.linspace(-195,-80)
#    thdot_num_first = np.where(result.t >= tnum[0])#t >= tnum[0])
#    thdot_num_last = np.where(result.t >= tnum[-1])#t >= tnum[-1])
  #  print('thdot_num_first[0][0] = '+str(thdot_num_first[0][0]))
  #  print('thdot_num_last = '+str(thdot_num_last))
    
#    numerator = integ.cumtrapz(thdarr[thdot_num_first[0][0]:thdot_num_last[0][0]],result.t[thdot_num_first[0][0]:thdot_num_last[0][0]])[-1]#t[thdot_num_first[0][0]:thdot_num_last[0][0]])[-1]
#    print('num / denom = '+str(numerator/denominator))
#    print('In this interval, N = '+str(Nconv[thdot_num_last[0][0]] - Nconv[thdot_num_first[0][0]]))
    
    
    
    
    fNL = 5./3*result.y[11][-1]/result.y[10][-1]**2
    gNL = 25./9*result.y[12][-1]/result.y[10][-1]**3

    ######  Find tau --------
    
    arg = []
    for i in range(len(result.t)):
        aIC = exp(result.y[-1][i])
        arg.append(1./aIC)
    tau = integ.cumtrapz(arg , result.t, 10)#conformal time    
    ####### End find tau ---------
    
    nu = 1.5
    amp = (-tau[-1])**(.5-nu)/sqrt(2) / delta_sstart
    As = 1/(2*pi**2)*result.y[10][-1]**2*amp**2
    #^Ignoring k^(3 - 2nu)
    logAs = log(As)

    tt = tau[-1]

    return [fNL,gNL,logAs,tt]



    