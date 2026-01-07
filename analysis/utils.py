from numpy import *
from numpy.fft import fft
def periodogram(data,dt,df):
     """
     data: long 1D array of stationary process, longer than NFFT=1./(dt*df)
     Optimal is data length NFFT*Ntrials, but doesn't have to be
    multiple of NFFT
     """
     NFFT=int(1./(dt*df)+0.5)
     df=1./(NFFT*dt)
     L=len(data)
     Ntrials=int(L/NFFT)
     x=data[:Ntrials*NFFT].reshape((-1,NFFT))
     ntrials=x.shape[0]
     xF=fft(x)
     S=sum(real(xF*xF.conjugate()),axis=0)*dt/(NFFT-1)/ntrials
     psd=S[1:int(NFFT/2)]
     freq=df*arange(NFFT/2-1)+df
     return (freq,psd)