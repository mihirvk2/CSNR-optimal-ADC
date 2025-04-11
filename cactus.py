import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scp
import scipy.special as sp
from tqdm import tqdm
from scipy import integrate
import math
from probability_distributions import *


def msce(N, pmf, t1, tM, B_ADC, sigma_ADC, delta_imc):
    # returns mean squared computational error for uniform quantization of pre-ADC IMC output
    M = 2**B_ADC - 1
    v_lsb = (tM - t1) / (M - 1)
    # Avoid division by zero for v_lsb and delta_imc
    if v_lsb == 0 or delta_imc == 0:
        raise ValueError("v_lsb and delta_imc must be non-zero values.")
    t = np.linspace(t1, tM, M)
    err_second_moment = 0
    for i in range(N + 1):
        temp_second_moment = 0
        for j in range(1, M + 1):
            # Check for valid input to normal_cdf
            argument = (t[j-1] - delta_imc * i) / sigma_ADC
            if not np.isfinite(argument):
                print(f"Warning: invalid argument for normal_cdf at i={i}, j={j}, argument={argument}")
                continue  # Skip this iteration if the argument is invalid
            temp_term = ((j-1+t1/v_lsb)*2*(v_lsb**2)/delta_imc**2 - 2*i*(v_lsb/delta_imc))
            temp_second_moment += temp_term * normal_cdf(argument)
        # Calculate num_second_moment
        num_second_moment = ((t1 + (M - 0.5)*v_lsb)/delta_imc - i)**2 - temp_second_moment
        err_second_moment += pmf[i] * num_second_moment
    return err_second_moment

def cactus(N, pmf, B_ADC, sigma_ADC, delta_imc):
    # compute SNR optimal ADC clipping thresholds search
    logN = np.log2(N)
    t1opt = 0
    tMopt = 0
    msce_min = np.inf
    M = 2**B_ADC - 1
    if(B_ADC >= logN):
        if(B_ADC > logN):
            print(f"B_ADC higher than optimal value. Choose B_ADC = {np.floor(logN)}")
        v_lsb = delta_imc
        t1opt = 0.5*v_lsb
        tMopt = (M-0.5)*v_lsb
        msce_min = msce(N, pmf, t1opt, tMopt, B_ADC, sigma_ADC, delta_imc)
        return [t1opt, tMopt], msce_min
    else:
        k = 1
        while((M-0.5)*k*delta_imc < delta_imc*N):
            v_lsb = k*delta_imc
            l = 0
            while((l+0.5)*delta_imc + (M-1)*v_lsb < delta_imc*N):
                t1_temp = (l+0.5)*delta_imc
                tM_temp = t1_temp + (M-1)*v_lsb
                msce_temp = msce(N, pmf, t1_temp, tM_temp, B_ADC, sigma_ADC, delta_imc)
                if(msce_temp < msce_min):
                    msce_min = msce_temp
                    t1opt = t1_temp
                    tMopt = tM_temp
                l+=1
            k+=1
        return [t1opt, tMopt], msce_min

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