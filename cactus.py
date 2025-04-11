import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scp
import scipy.special as sp
from tqdm import tqdm
from scipy import integrate
import math
from probability_distributions import *
from mse_dp import *

def cactus(N, pmf, B_ADC, sigma_ADC, delta_imc):
    # compute SNR optimal ADC clipping thresholds search
    logN = np.log2(N)
    t1opt = 0
    tMopt = 0
    mse_dp_min = np.inf
    M = 2**B_ADC - 1
    if(B_ADC >= logN):
        if(B_ADC > logN):
            print(f"B_ADC higher than optimal value. Choose B_ADC = {np.floor(logN)}")
        v_lsb = delta_imc
        t1opt = 0.5*v_lsb
        tMopt = (M-0.5)*v_lsb
        mse_dp_min, _ = mse_dp(N, pmf, t1opt, tMopt, B_ADC, sigma_ADC, delta_imc)
        return [t1opt, tMopt], mse_dp_min
    else:
        k = 1
        while((M-0.5)*k*delta_imc < delta_imc*N):
            v_lsb = k*delta_imc
            l = 0
            while((l+0.5)*delta_imc + (M-1)*v_lsb < delta_imc*N):
                t1_temp = (l+0.5)*delta_imc
                tM_temp = t1_temp + (M-1)*v_lsb
                mse_dp_temp, _ = mse_dp(N, pmf, t1_temp, tM_temp, B_ADC, sigma_ADC, delta_imc)
                if(mse_dp_temp < mse_dp_min):
                    mse_dp_min = mse_dp_temp
                    t1opt = t1_temp
                    tMopt = tM_temp
                l+=1
            k+=1
        return [t1opt, tMopt], mse_dp_min

if __name__ == '__main__':
    # Example usage
    p = 0.25
    N = 16
    pmf,_,_ = binomial_pmf(N, p)
    sigma_ADC = 0.005
    VDD = 0.9
    delta_imc = VDD / (N * 1.3)
    B_ADC = 3
    
    t_clipping = cactus(N, pmf, B_ADC, sigma_ADC, delta_imc)[0]
    print(np.array(t_clipping)/delta_imc)  