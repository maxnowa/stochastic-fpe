import numpy as np
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
     S=np.sum(np.real(xF*xF.conjugate()),axis=0)*dt/(NFFT-1)/ntrials
     psd=S[1:int(NFFT/2)]
     freq=df*np.arange(int(NFFT/2)-1)+df
     return (freq,psd)

def log_smooth(freq, psd, bins_per_decade=20):
    """
    Smooths the PSD by averaging data points into logarithmic frequency bins.
    This creates a clean line on a log-log plot while preserving the total power.
    """
    # Create log-spaced bins from min to max frequency
    min_f = freq[1]  # Skip DC component (freq=0)
    max_f = freq[-1]
    
    # Generate bin edges
    bin_edges = np.logspace(np.log10(min_f), np.log10(max_f), 
                            int(np.log10(max_f/min_f) * bins_per_decade))
    
    # Digitize the frequencies into these bins
    bin_indices = np.digitize(freq, bin_edges)
    
    # Calculate the mean Frequency and PSD for each bin
    smooth_freq = []
    smooth_psd = []
    
    # Iterate through bins (using a fast vectorized approach would be better for massive data, 
    # but this loop is safe and clear for standard usage)
    for i in range(1, len(bin_edges)):
        mask = bin_indices == i
        if np.any(mask):
            smooth_freq.append(np.mean(freq[mask]))
            smooth_psd.append(np.mean(psd[mask]))
            
    return np.array(smooth_freq), np.array(smooth_psd)