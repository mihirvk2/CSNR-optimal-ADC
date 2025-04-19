import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import seaborn as sns
import scipy as scp
import scipy.special as sp
from tqdm import tqdm
from scipy import integrate
from probability_distributions import *
import math
import logging
from scipy.interpolate import griddata


def mse_dp(N, pmf, t1, tM, b_adc, sigma_adc, delta_imc):
    # Computes the mean squared computational error (MSE_dp) due to uniform quantization of IMC pre-ADC output
    # Returns the mse_dp, csnr and a boolean indicating valid computation

    M = 2**b_adc - 1
    v_lsb = (tM - t1) / (M - 1)
    # Avoid division by zero for v_lsb and delta_imc
    if v_lsb == 0 or delta_imc == 0:
        print("v_lsb and delta_imc must be non-zero values.")
        return np.inf, False
        # raise ValueError("v_lsb and delta_imc must be non-zero values.")
    t = np.linspace(t1, tM, M)
    alpha = 0
    e_offset = 0
    y_first_mom = 0
    y_second_mom = 0
    for i in range(N + 1):
        temp_alpha = 0
        temp_e_offset = 0
        y_first_mom += i*pmf[i]
        y_second_mom += (i**2)*pmf[i]
        for j in range(1, M + 1):
            # Check for valid input to normal_cdf
            argument = (t[j-1] - i*delta_imc) / sigma_adc
            if not np.isfinite(argument):
                print(f"Warning: invalid argument for normal_cdf at i={i}, j={j}, argument={argument}")
                continue  # Skip this iteration if the argument is invalid
            temp_alpha += (t[j-1]- i*delta_imc)* normal_cdf(argument)
            temp_e_offset += normal_cdf(argument)
        # Calculate MSE_dp
        inner_exp_i_alpha  = ((t[M-1] + 0.5*v_lsb - i*delta_imc)**2 - 2*v_lsb*temp_alpha)/ delta_imc**2 
        inner_exp_i_e_offset = (t[M-1] + 0.5*v_lsb - i*delta_imc - v_lsb*temp_e_offset)/ delta_imc 
        alpha += pmf[i]*inner_exp_i_alpha 
        e_offset += pmf[i]*inner_exp_i_e_offset
    MSE_dp = alpha - e_offset**2
    y_var = y_second_mom - y_first_mom**2
    CSNR = y_var/MSE_dp

    return MSE_dp, CSNR, True



def mse_dp_thresholds_sweep(N, pmf, b_adc, sigma_adc, delta_imc, npoints):
    # Sweeps across different threshold ranges to compute MSE_dp and stores the results for analysis and visualization

    thresholds = np.linspace(0,N*delta_imc,npoints)
    t1 = []
    tM = []
    mse_dp_val = []
    logging.basicConfig(filename=f'Outputs/mse_dp_thresholds_sweep_N_{N}_b_acd_{b_adc}_sigma_adc_{sigma_adc}_delta_imc_{delta_imc}_new.log', level=logging.INFO, filemode='w')
    for i in tqdm(range(npoints)):
        t1_cur = thresholds[i]
        for j in tqdm(range(i+1,npoints)): 
            tM_cur = thresholds[j]
            mse_dp_val_cur, csnr_val_cur, valid = mse_dp(N, pmf, t1_cur, tM_cur, b_adc, sigma_adc, delta_imc)
            if(valid):
                t1.append(t1_cur)
                tM.append(tM_cur)
                mse_dp_val.append(mse_dp_val_cur)
                logging.info(f"For t1 = {t1_cur} and tM = {tM_cur}, mse_dp = {mse_dp_val_cur}")

    t1 = np.array(t1)
    tM = np.array(tM)
    mse_dp_val = np.array(mse_dp_val)

    np.save(f'Outputs/mse_dp_thresholds_sweep_t1_N_{N}_b_acd_{b_adc}_sigma_adc_{sigma_adc}_delta_imc_{delta_imc}_new.npy', t1)
    np.save(f'Outputs/mse_dp_thresholds_sweep_tM_N_{N}_b_acd_{b_adc}_sigma_adc_{sigma_adc}_delta_imc_{delta_imc}_new.npy', tM)
    np.save(f'Outputs/mse_dp_thresholds_mse_dp_val_N_{N}_b_acd_{b_adc}_sigma_adc_{sigma_adc}_delta_imc_{delta_imc}_new.npy', mse_dp_val)

def plot_mse_dp(N, b_adc, sigma_adc, delta_imc):
    # Visualizes the MSE_dp sweep results as a heatmap over the (t1, tM) threshold grid

    t1 = np.load(f'Outputs/mse_dp_thresholds_sweep_t1_N_{N}_b_acd_{b_adc}_sigma_adc_{sigma_adc}_delta_imc_{delta_imc}_new.npy')
    tM = np.load(f'Outputs/mse_dp_thresholds_sweep_tM_N_{N}_b_acd_{b_adc}_sigma_adc_{sigma_adc}_delta_imc_{delta_imc}_new.npy')
    mse_dp_val = np.load(f'Outputs/mse_dp_thresholds_mse_dp_val_N_{N}_b_acd_{b_adc}_sigma_adc_{sigma_adc}_delta_imc_{delta_imc}_new.npy')

    npoints = len(t1)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Scatter plot
    sc = ax.scatter(t1/delta_imc, tM/delta_imc, c=10*np.log10(N*0.25*0.75) - 10*np.log10(mse_dp_val), cmap='viridis', s=20, alpha=0.8, vmin=0, vmax=50)

    # Labels
    ax.set_xlabel(r'$t_1/\Delta_\mathrm{imc}$', fontsize = 13)
    ax.set_ylabel(r'$t_M/\Delta_\mathrm{imc}$', fontsize = 13)
    # ax.set_title('Mean squared error (dB)', fontsize = 12)

    # ax.set_xlim((0.04,0.14))
    # ax.set_ylim((0.2,0.3))
    ax.set_xlim((0,24))
    ax.set_ylim((64,88))
    # ax.set_xticks(range(0, N+1, N//8))
    # ax.set_yticks(range(0, N+1, N//8))
    # Colorbar
    fig.colorbar(sc, label=r'$\mathrm{CSNR}$ (dB)', shrink=0.8, aspect=10)
    plt.savefig(f'Figures/zoomed_csnr_thresholds_sweep_N_{N}_b_acd_{b_adc}_sigma_adc_{sigma_adc}_delta_imc_{delta_imc}_new.png', format='png', bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    # Example usage

    N = 128
    p = 0.25
    pmf, sig_mean, sig_var = binomial_pmf(N, p)
    delta_imc = 0.005 # 5 mV
    b_adc = int(np.ceil(np.log2(N))-1)
    # b_adc = int(np.ceil(np.log2(N))-1)
    sigma_adc = delta_imc/10
    npoints = N*25
    # mse_dp_thresholds_sweep(N, pmf, b_adc, sigma_adc, delta_imc, npoints)
    plot_mse_dp(N, b_adc, sigma_adc, delta_imc)
    # mse_dp_thresholds_sweep(N, pmf, b_adc-1, sigma_adc, delta_imc, npoints)    
    # plot_mse_dp(N, b_adc-1, sigma_adc, delta_imc)