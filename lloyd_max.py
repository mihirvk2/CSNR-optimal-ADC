import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scp
import scipy.special as sp
from tqdm import tqdm
from scipy import integrate
import math
from probability_distributions import *

def lloyd_max_gaussian(N, p, B_ADC, delta_imc, niter):
    # Performs Lloyd-Max quantiation assuming Gaussian-distributed dot product result
    # Returns optimal thresholds and representation levels normalized by delta_imc

    # Initialize variables
    M = 2**B_ADC - 1
    
    # Initial conditions
    r_vec_ana = np.linspace(0, N*delta_imc*2*p, M + 1)
    t_vec = (r_vec_ana[:-1] + r_vec_ana[1:]) / 2
    
    for i in tqdm(range(niter)):
        # Update representation points
        for j in range(M + 1):
            if j == 0:
                P_Y_in_U, _ = integrate.quad(gaussian_pdf, -np.inf, t_vec[0], args=(N*p*delta_imc, np.sqrt(N*p*(1-p))*delta_imc))
                Y_mean, _ = integrate.quad(x_times_p_x_gaussian, -np.inf, t_vec[0], args=(N, p, delta_imc))
                if P_Y_in_U != 0:
                    r_vec_ana[0] = Y_mean / P_Y_in_U 
            elif j == M:
                P_Y_in_U, _ = integrate.quad(gaussian_pdf, t_vec[M - 1], np.inf, args=(N*p*delta_imc, np.sqrt(N*p*(1-p))*delta_imc))
                Y_mean, _ = integrate.quad(x_times_p_x_gaussian, t_vec[M - 1], np.inf, args=(N, p, delta_imc))
                if P_Y_in_U != 0:
                    r_vec_ana[M] = Y_mean / P_Y_in_U
            else:
                P_Y_in_U, _ = integrate.quad(gaussian_pdf, t_vec[j - 1], t_vec[j], args=(N*p*delta_imc, np.sqrt(N*p*(1-p))*delta_imc))
                Y_mean, _ = integrate.quad(x_times_p_x_gaussian, t_vec[j - 1], t_vec[j], args=(N, p, delta_imc))
                if P_Y_in_U != 0:
                    r_vec_ana[j] = Y_mean / P_Y_in_U
        # Update thresholds
        t_vec = (r_vec_ana[:-1] + r_vec_ana[1:]) / 2 

    return t_vec, r_vec_ana/delta_imc


if __name__ == '__main__':
    # Example usage
    p = 0.25
    N = 16
    VDD = 0.9
    delta_imc = VDD / (N * 1.3)
    B_ADC = 3
    niter = 1000
    t, r = lloyd_max_gaussian(N, p, B_ADC, delta_imc, niter)
    print(r)  