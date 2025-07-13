import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.stats import norm
from scipy.integrate import quad
from tqdm import tqdm
import logging

def mse_q_gaussian(N, p, t1, tM, b_adc, delta_imc):
    # Computes the mean squared quantization error (MSE_q) for uniform quantization of a Gaussian-distributed IMC output
    # Returns the MSE_q and a boolean indicating successful computation

    M = 2**b_adc - 1  # number of thresholds
    if M <= 1:
        print("Invalid number of thresholds computed from b_adc.")
        return np.inf, False
    
    v_lsb = (tM - t1) / (M - 1)
    if v_lsb == 0 or delta_imc == 0:
        print("v_lsb and delta_imc must be non-zero values.")
        return np.inf, False
    
    # Gaussian parameters
    mu = N *p * delta_imc
    sigma2 = N * p * (1-p) * delta_imc**2
    sigma = np.sqrt(sigma2)
    
    # Define the thresholds for the bins (uniformly spaced)
    thresholds = np.linspace(t1, tM, M)
    
    mse_q = 0.0
    # Compute the error for each quantization bin (M-1 bins)
    for i in range(M - 1):
        bin_low = thresholds[i]
        bin_high = thresholds[i+1]
        rep = (bin_low + bin_high) / 2.0  # representation (quantization) point
        
        # Integrate the squared error over the bin
        error, _ = quad(lambda x: (x - rep)**2 * norm.pdf(x, loc=mu, scale=sigma),
                        bin_low, bin_high)
        mse_q += error

    # Handle saturation: for values outside the thresholds, assume saturation at the extreme representation points.
    rep_low = t1 - v_lsb/2.0
    rep_high = tM + v_lsb/2.0
    
    error_left, _ = quad(lambda x: (x - rep_low)**2 * norm.pdf(x, loc=mu, scale=sigma),
                           -np.inf, t1)
    error_right, _ = quad(lambda x: (x - rep_high)**2 * norm.pdf(x, loc=mu, scale=sigma),
                           tM, np.inf)
    mse_q += error_left + error_right
    
    return mse_q, True

def mse_q_thresholds_sweep_gaussian(N, p, b_adc, delta_imc, npoints):
    # Sweeps across different threshold ranges to compute MSE_q and stores the results for analysis and visualization

    thresholds_range = np.linspace(0, N * delta_imc, npoints)
    
    t1_list = []
    tM_list = []
    mse_q_val_list = []
    
    # Logging configuration
    logging.basicConfig(filename=f'Outputs/mse_q_thresholds_sweep_N_{N}_b_adc_{b_adc}_delta_imc_{delta_imc}.log',
                        level=logging.INFO, filemode='w')
    
    # Sweep over possible (t1, tM) pairs with tM > t1
    for i in tqdm(range(npoints), desc="Sweeping t1"):
        t1_cur = thresholds_range[i]
        for j in tqdm(range(i+1, npoints), desc="Sweeping tM", leave=False):
            tM_cur = thresholds_range[j]
            mse_q_val_cur, valid = mse_q_gaussian(N, p, t1_cur, tM_cur, b_adc, delta_imc)
            if valid:
                t1_list.append(t1_cur)
                tM_list.append(tM_cur)
                mse_q_val_list.append(mse_q_val_cur)
                logging.info(f"For t1 = {t1_cur} and tM = {tM_cur}, mse_q = {mse_q_val_cur}")
    
    t1_array = np.array(t1_list)
    tM_array = np.array(tM_list)
    mse_q_array = np.array(mse_q_val_list)
    
    # Save the data
    np.save(f'Outputs/mse_q_thresholds_sweep_t1_N_{N}_b_adc_{b_adc}_delta_imc_{delta_imc}.npy', t1_array)
    np.save(f'Outputs/mse_q_thresholds_sweep_tM_N_{N}_b_adc_{b_adc}_delta_imc_{delta_imc}.npy', tM_array)
    np.save(f'Outputs/mse_q_thresholds_mse_q_val_N_{N}_b_adc_{b_adc}_delta_imc_{delta_imc}.npy', mse_q_array)
    

def plot_mse_q(N, b_adc, delta_imc):
    # Visualizes the MSE_q sweep results as a heatmap over the (t1, tM) threshold grid

    t1 = np.load(f'Outputs/mse_q_thresholds_sweep_t1_N_{N}_b_adc_{b_adc}_delta_imc_{delta_imc}.npy')
    tM = np.load(f'Outputs/mse_q_thresholds_sweep_tM_N_{N}_b_adc_{b_adc}_delta_imc_{delta_imc}.npy')
    mse_q_val = np.load(f'Outputs/mse_q_thresholds_mse_q_val_N_{N}_b_adc_{b_adc}_delta_imc_{delta_imc}.npy')
    mse_q_val_normalized = mse_q_val/(delta_imc**2)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    sc = ax.scatter(t1, tM, c=10 * np.log10(mse_q_val_normalized), cmap='viridis', s=20, alpha=0.8, vmin=-30, vmax=30)

    # Labels
    ax.set_xlabel(r'$t_1$ (V)', fontsize = 13)
    ax.set_ylabel(r'$t_M$ (V)', fontsize = 13)

    # Colorbar
    fig.colorbar(sc, label=r'$\mathrm{MSE}_\mathrm{q}$ (dB)', shrink=0.8, aspect=10)
    plt.savefig(f'Figures/mse_q_thresholds_sweep_N_{N}_b_acd_{b_adc}_delta_imc_{delta_imc}.png', format='png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # Example usage

    N = 128
    p = 0.25
    v_dd = 0.9
    delta_imc = v_dd / (N*1.3)
    b_adc = int(np.ceil(np.log2(N))-2)
    # b_adc = int(np.ceil(np.log2(N))-1)
    npoints = N*2
    # mse_q_thresholds_sweep_gaussian(N, p, b_adc, delta_imc, npoints)
    # mse_q_thresholds_sweep_gaussian(N, p, b_adc-1, delta_imc, npoints)
    plot_mse_q(N, b_adc, delta_imc)

