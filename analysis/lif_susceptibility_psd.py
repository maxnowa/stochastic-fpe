from pylab import *
from mpmath import pcfu,gamma
import numpy as np # Added for get_theoretical_curve

def U_func1(a,z):
    if (z>10.):
        return exp(-0.25* z**2)*z**(-a-0.5)*(1 - (a+0.5)*(a+1.5)/(2*z**2) + (a+0.5)*(a+1.5)*(a+2.5)*(a+3.5)/(8*z**4))
    elif (z<-10):
        return sqrt(2*pi)*exp(0.25 * z**2)*abs(z)**(a-0.5)/gamma(0.5+a)*(1 + (a-0.5)*(a-1.5)/(2*z**2) + (a-0.5)*(a-1.5)*(a-2.5)*(a-3.5)/(8*z**4)) - sin(pi*a)*exp(-0.25 * z**2)*abs(z)**(-a-0.5)*(1 - (a+0.5)*(a+1.5)/(2*z**2) + (a+0.5)*(a+1.5)*(a+2.5)*(a+3.5)/(8*z**4))
    else:
        return pcfu(a,z)

def U_func2(a,z,delta):
    if (z>10.):
        A=exp(delta-0.25* z**2)
        return A*z**(-a-0.5)*(1 - (a+0.5)*(a+1.5)/(2*z**2) + (a+0.5)*(a+1.5)*(a+2.5)*(a+3.5)/(8*z**4))
    elif (z<-10):
        A=exp(delta+0.25 * z**2)
        return sqrt(2*pi)*A*abs(z)**(a-0.5)/gamma(0.5+a)*(1 + (a-0.5)*(a-1.5)/(2*z**2) + (a-0.5)*(a-1.5)*(a-2.5)*(a-3.5)/(8*z**4)) - sin(pi*a)*exp(-0.25 * z**2)*abs(z)**(-a-0.5)*(1 - (a+0.5)*(a+1.5)/(2*z**2) + (a+0.5)*(a+1.5)*(a+2.5)*(a+3.5)/(8*z**4))
    else:
        return exp(delta)*pcfu(a,z)

def psd_lif_normalized(om,mu,D,vth,vr,tref):
    delta=(vr**2-vth**2+2*mu*(vth-vr))/(4*D)
    a=-1j*om-0.5
    Dvth=U_func1(a,(mu-vth)/sqrt(D))
    Dvr=U_func2(a,(mu-vr)/sqrt(D),delta)
    return (abs(Dvth)**2-abs(Dvr)**2)/abs(Dvth-exp(1j*om*tref)*Dvr)**2

def suscept_lif_normalized(om,mu,D,vth,vr,tref):
    delta=(vr**2-vth**2+2*mu*(vth-vr))/(4*D)
    a=-1j*om-0.5
    z=(mu-vth)/sqrt(D)
    Dvth=U_func1(a,z)
    Dvth1=U_func1(a+1,z)
    z=(mu-vr)/sqrt(D)
    Dvr=U_func2(a,z,delta)
    Dvr1=U_func2(a+1,z,delta)
    return complex(1j*om/sqrt(D)/(1j*om-1)*(Dvth1-Dvr1)/(Dvth-exp(1j*om*tref)*Dvr))

def get_theoretical_curve(freqs_hz, mu, D, tau, V_th, V_reset, r0_hz, N):
    """
    Calculates the theoretical PSD curve for the Finite-Size LIF neuron.
    """
    
    # 1. Non-dimensionalize parameters
    theta = V_th - V_reset
    mu_dim = (mu - V_reset) / theta
    D_dim = D / (theta**2)
    
    # Dimensionless boundaries
    vth_dim = 1.0
    vr_dim = 0.0
    tref_dim = 0.0 
    
    psd_vals = []
    scaling_factor = r0_hz / N  
    
    for f in freqs_hz:
        if f == 0:
            psd_vals.append(np.nan)
            continue
            
        # om = (2 * pi * f_hz) * tau_sec
        om_dim = 2 * np.pi * f * (tau / 1000.0)
        
        try:
            norm_val = psd_lif_normalized(om_dim, mu_dim, D_dim, vth_dim, vr_dim, tref_dim)
            psd_vals.append(float(abs(norm_val)) * scaling_factor)
        except Exception:
            psd_vals.append(np.nan)

    print("Done.")
    return np.array(psd_vals)