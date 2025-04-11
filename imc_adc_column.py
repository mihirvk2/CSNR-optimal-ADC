import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from adc import uniform_adc_imc
from adc import non_uniform_adc_imc
from helper_functions import *
from cactus import *
from lloyd_max import *
import scipy as scp
import scipy.special as sp
import logging

class qr_bpbs_dp_unit:
    # Dot product unit for QR SRAM-based In-Memory Computing with BPBS (Bit-Parallel weight Bit-Serial input) scheme

    # Parameters:
    # n_phy_rows     : Number of physical rows in each IMC column
    # b_w            : Bit precision of weights
    # b_x            : Bit precision of input activations
    # b_adc          : Bit precision of ADCs
    # sigma_adc      : Standard deviation of ADC thermal noise
    # c_qr_mean      : Mean unit cell capacitance (in femtofarads, fF)
    # c_qr_sigma     : Standard deviation of unit cell capacitance, derived from c_qr_mean
    # c_qr_array     : Matrix representing actual cell capacitances, sampled from a normal distribution
    # c_par          : Parasitic capacitance on the capacitance line (depends on c_qr_mean and n_phy_rows)
    # v_dd           : Supply voltage (default 0.9V)
    # delta_imc      : Voltage spacing between adjacent pre-ADC dot product levels
    # v_cl           : Array of possible capacitance line voltages
    # w_array        : Weight matrix (bit-parallel form), initially all zeros
    # w_scale        : Weight scaling factor for normalization
    # analog_pots    : Boolean flag to enable analog power-of-two-summing (not implemented)
    # heterogenous_adc : Boolean flag for enabling different ADC precision per column (not implemented)
    # adc_config     : Mapping of ADC precision to the corresponding ADC index
    # n_adc          : Total number of unique ADCs based on adc_config
    # adc_array      : List of ADC instances (uniform_adc_imc), each configured with thresholds and noise.

    # Note:
    # - Analog POTS and heterogeneous precision ADCs are not implemented in this work.

    def __init__(self, n_phy_rows, b_w, b_x, b_adc, sigma_adc, c_qr_mean=1, v_dd=0.9, analog_pots = False, heterogenous_adc = False, adc_config = None):
        # Instantiate IMC DP unit with all the necessary parameters
        self.n_phy_rows = n_phy_rows
        self.b_w = b_w
        self.b_x = b_x
        self.b_adc = b_adc
        self.sigma_adc = sigma_adc
        self.c_qr_mean = c_qr_mean
        self.c_qr_sigma = 2.1*10**(-2.5)*np.sqrt(c_qr_mean) # k-model value for 28nm technology
        self.c_qr_array = np.random.normal(self.c_qr_mean, self.c_qr_sigma, (self.n_phy_rows, self.b_w))
        self.w_scale = 1
        self.w_array = np.zeros((self.n_phy_rows, self.b_w))
        self.v_dd = 0.9
        self.c_par = self.c_qr_mean*self.n_phy_rows*0.3 + 2.04278 
        self.analog_pots = analog_pots
        self.heterogenous_adc = heterogenous_adc
        self.delta_imc = self.c_qr_mean*self.v_dd/(self.n_phy_rows*self.c_qr_mean + self.c_par)
        self.v_cl = np.arange(self.n_phy_rows+1)*self.delta_imc
        if self.analog_pots == False:
            # [0, 1, 2, 3] : ADC_0 for column 0, ADC_1 for column 1, ...
            self.adc_config = np.array(range(b_w)) 
        else:
            if adc_config is None: 
                print("ADC configuration not provided")
                return
            else:
                # can be something like [0, 1, 2, 2] : ADC_0 for col 0, ADC_1 for col 1, ADC_2 for col 2 and 3
                self.adc_config = adc_config
                self.n_adc =  len(set(self.adc_config))
        self.n_adc = len(self.adc_config)
        if self.heterogenous_adc == False:
            self.adc_array = [uniform_adc_imc(b_adc=b_adc, t1=0.5*self.delta_imc*self.n_phy_rows/(2**b_adc), tM = (2**b_adc - 1.5)*self.delta_imc*self.n_phy_rows/(2**b_adc), sigma_adc=self.sigma_adc, v_cl= self.v_cl) for i in range(self.n_adc)]
        else:
            self.adc_array = [uniform_adc_imc(b_adc=b_adc[i], t1=0.5*self.delta_imc*self.n_phy_rows/(2**b_adc[i]), tM = (2**b_adc[i] - 1.5)*self.delta_imc*self.n_phy_rows/(2**b_adc[i]), sigma_adc=self.sigma_adc, v_cl= self.v_cl) for i in range(self.n_adc)]

    def pots(self, x):
        # Power of two summation
        x_len = len(x)
        x_pots = 0
        for i in range(x_len):
            if(i==0):
                x_pots -= x[i]*2**(x_len-1)
            else:
                x_pots += x[i]*2**(x_len-i-1)
        return x_pots
    
    def store_weights(self, w_vec, w_range):
        # Store weights in memory assuming uniform quantization in the range [-w_range, w_range)
        w_vec_int, self.w_scale = quantize_to_signed_int(w_vec, self.b_w, w_range)
        w_vec_bits = int_to_bits(w_vec_int, self.b_w)
        w_rows = w_vec_bits.shape[0]
        w_cols = self.b_w 
        if(w_rows>self.n_phy_rows):
            print("Weight vector cannot be mapped in this IMC dot product unit")
        else:
            for i in range(w_rows):
                for j in range(w_cols):
                    self.w_array[i,j] =  w_vec_bits[i,j]
        return
    
    def compute_dp(self, x_vec, x_range):
        # Computes 2's complement dot product between input activation and stored weights
        # Assumes uniform quantization for input activations in the range [-x_range, x_range)
        dp_dim = len(x_vec)
        if(dp_dim>self.n_phy_rows): 
            print("Size of input vector larger than number of IMC rows")
            return
        x_vec_int, x_scale = quantize_to_signed_int(x_vec, self.b_x, x_range)
        x_vec_bits = int_to_bits(x_vec_int, self.b_x)
        bpbs_pots_result = 0
        col_pots_result = np.zeros(self.b_x)
        for k in range(self.b_x):
            v_cl = np.zeros(self.b_w)
            r_points = np.zeros(self.n_adc)
            for j in range(self.b_w):
                cap_tot = 0
                sum = 0 
                for i in range(self.n_phy_rows):
                    cap_tot += self.c_qr_array[i,j]
                    if(i<dp_dim):
                        sum += self.w_array[i,j]*x_vec_bits[i,k]*self.c_qr_array[i,j]
                cap_tot += self.c_par
                v_cl[j] = sum*self.v_dd/cap_tot
                r_points[j] = self.adc_array[j].convert(v_cl[j])
            col_pots_result[k] = self.pots(r_points)
        bpbs_pots_result = self.pots(col_pots_result)
        return bpbs_pots_result*x_scale*self.w_scale
    
    def compute_snr(self, n_samples):
        # Computes SNR of the IMC with respect to fixed point baseline
        y_ideal = np.zeros(n_samples)
        y_imc = np.zeros(n_samples)
        n_errors = 0
        tolerance = 1e-10 
        for i in tqdm(range(n_samples)):
            w_vec = np.random.randint(-2**(self.b_w-1),2**(self.b_w-1),size=self.n_phy_rows)
            x_vec = np.random.randint(-2**(self.b_x-1),2**(self.b_x-1),size=self.n_phy_rows)
            y_ideal[i] = np.dot(w_vec, x_vec)
            self.store_weights(w_vec, 2**(self.b_w-1))
            y_imc[i] = self.compute_dp(x_vec, 2**(self.b_x-1))
            # Fix floating point errors: if y_imc is within tolerance of y_ideal, update it
            if abs(y_imc[i] - y_ideal[i]) < tolerance:
                y_imc[i] = y_ideal[i]
            if(y_imc[i]!=y_ideal[i]):
                n_errors += 1
        print(f"Number of errors = {n_errors}")
        logging.info(f"Number of errors = {n_errors}")
        if(n_errors==0):
            print(f"NO ERRORS DETECTED IN {n_samples} SAMPLES")
            logging.info(f"NO ERRORS DETECTED IN {n_samples} SAMPLES")
            return 100 # large number
        return 10*np.log10(np.var(y_ideal)/np.var(y_ideal-y_imc))
        
    
    def update_c_qr(self, c_qr_mean_new):
        # Updates c_qr and all the other parameters derived from it
        self.c_qr_mean = c_qr_mean_new
        self.c_qr_sigma = 2.1*10**(-2.5)*np.sqrt(c_qr_mean_new) # k-model value for 28nm technology
        self.c_par = self.c_qr_mean*self.n_phy_rows*0.3 + 2.04278
        self.c_qr_array = np.random.normal(self.c_qr_mean, self.c_qr_sigma, (self.n_phy_rows, self.b_w))
        self.delta_imc = self.c_qr_mean*self.v_dd/(self.n_phy_rows*self.c_qr_mean + self.c_par)
        self.v_cl = self.delta_imc/2 + np.arange(self.n_phy_rows+1)*self.delta_imc
        for i in range(self.n_adc):
            self.adc_array[i].v_cl = self.v_cl
            self.adc_array[i].t1 =  0.5*self.delta_imc*self.n_phy_rows/(2**self.adc_array[i].b_adc) 
            self.adc_array[i].tM = (2**self.adc_array[i].b_adc - 1.5)*self.delta_imc*self.n_phy_rows/(2**self.adc_array[i].b_adc)
            self.adc_array[i].t_vec = np.linspace(self.adc_array[i].t1, self.adc_array[i].tM, self.adc_array[i].M)
            self.adc_array[i].v_lsb = (self.adc_array[i].tM-self.adc_array[i].t1)/(self.adc_array[i].M-1)
            self.adc_array[i].r_vec = np.zeros(self.adc_array[i].M+1)
            self.adc_array[i].r_vec[0] = (self.adc_array[i].t1 - self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
            self.adc_array[i].r_vec[1:] = (self.adc_array[i].t_vec + self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
        return
    
    def update_b_adc(self, b_adc):
        # Updates b_adc and all the other parameters derived from it
        for i in range(self.n_adc):
            self.adc_array[i].b_adc = b_adc
            self.adc_array[i].M = 2**self.adc_array[i].b_adc -1
            self.adc_array[i].t1 =  0.5*self.delta_imc*self.n_phy_rows/(2**self.adc_array[i].b_adc) 
            self.adc_array[i].tM = (2**self.adc_array[i].b_adc- 1.5)*self.delta_imc*self.n_phy_rows/(2**self.adc_array[i].b_adc)
            self.adc_array[i].t_vec = np.linspace(self.adc_array[i].t1, self.adc_array[i].tM, self.adc_array[i].M)
            self.adc_array[i].v_lsb = (self.adc_array[i].tM-self.adc_array[i].t1)/(self.adc_array[i].M-1)
            self.adc_array[i].r_vec = np.zeros(self.adc_array[i].M+1)
            self.adc_array[i].r_vec[0] = (self.adc_array[i].t1 - self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
            self.adc_array[i].r_vec[1:] = (self.adc_array[i].t_vec + self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
        return

    def update_sigma_adc(self, sigma_adc):
        # Updates sigma_adc and all the other parameters derived from it
        for i in range(self.n_adc):
            self.adc_array[i].sigma_adc = sigma_adc
        return

    def clip_adc_occ_gaussian(self, b_adc):
        # Assuming binomial pmf
        k_occ = 0
        N = self.w_array.shape[0]
        mu = N*0.25
        sigma = np.sqrt(N*0.25*0.75)
        # Numbers from Charbel's paper
        if(b_adc==2):
            k_occ = 1.71
        elif(b_adc==3):
            k_occ = 2.15
        elif(b_adc==4):
            k_occ = 2.55
        elif(b_adc==5):
            k_occ = 2.94
        elif(b_adc==6):
            k_occ = 3.29
        elif(b_adc==7):
            k_occ = 3.61
        elif(b_adc==8):
            k_occ = 3.92
        elif(b_adc==9):
            k_occ = 4.21
        elif(b_adc==10):
            k_occ = 4.49
        dig_ref_l = mu - k_occ*sigma
        dig_ref_h = mu + k_occ*sigma
        for i in range(self.n_adc):
            self.adc_array[i].b_adc = b_adc
            self.adc_array[i].M = 2**self.adc_array[i].b_adc -1
            self.adc_array[i].t1 = dig_ref_l*self.delta_imc
            self.adc_array[i].tM = dig_ref_h*self.delta_imc
            self.adc_array[i].t_vec = np.linspace(self.adc_array[i].t1, self.adc_array[i].tM, self.adc_array[i].M)
            self.adc_array[i].v_lsb = (self.adc_array[i].tM-self.adc_array[i].t1)/(self.adc_array[i].M-1)
            self.adc_array[i].r_vec = np.zeros(self.adc_array[i].M+1)
            self.adc_array[i].r_vec[0] = (self.adc_array[i].t1 -  self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
            self.adc_array[i].r_vec[1:] = (self.adc_array[i].t_vec +  self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
        return
    
    def clip_adc_lloyd_max_gaussian(self, b_adc, niter):
        # Assuming binomial pmf
        N = self.w_array.shape[0]
        p = 0.25
        t_vec, r_vec = lloyd_max_gaussian(N, p, b_adc, self.sigma_adc, self.delta_imc, niter)
        self.adc_array = [non_uniform_adc_imc(b_adc=b_adc, t_vec = t_vec, r_vec = r_vec, sigma_adc=self.sigma_adc, v_cl= self.v_cl) for i in range(self.n_adc)]

    def clip_adc_cactus(self, b_adc):
        # Assuming binomial pmf
        N = self.w_array.shape[0]
        pmf = np.zeros(N+1)
        for i in range(N+1):
            pmf[i] = scp.special.comb(N,i)*(0.25**i)*(1-0.25)**(N-i)
        t_opt, msce = cactus(N, pmf, b_adc, self.sigma_adc, self.delta_imc)
        for i in range(self.n_adc):
            self.adc_array[i].b_adc = b_adc
            self.adc_array[i].M = 2**self.adc_array[i].b_adc -1
            self.adc_array[i].t1 = t_opt[0]
            self.adc_array[i].tM = t_opt[1]
            self.adc_array[i].t_vec = np.linspace(self.adc_array[i].t1, self.adc_array[i].tM, self.adc_array[i].M)
            self.adc_array[i].v_lsb = (self.adc_array[i].tM-self.adc_array[i].t1)/(self.adc_array[i].M-1)
            self.adc_array[i].r_vec = np.zeros(self.adc_array[i].M+1)
            self.adc_array[i].r_vec[0] = (self.adc_array[i].t1 - self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
            self.adc_array[i].r_vec[1:] = (self.adc_array[i].t_vec +  self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
        return

def run_simulation(dp_dim, sigma_adc, npoints, date):
    # Runs compute SNR simulations across ADC bit precisions for various clipping strategies

    print(f"Running simulation for dp_dim = {dp_dim}, sigma_adc = {sigma_adc}, npoints = {npoints}, date = {date}")
    logging.basicConfig(filename=f'Outputs/imc_snr_b_adc_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.log', level=logging.INFO, filemode='w')

    b_adc_options = np.array(range(3, 10))
    b_adc_npoints = len(b_adc_options)

    compute_snr_fr = np.zeros(b_adc_npoints) # Full range 
    compute_snr_occ = np.zeros(b_adc_npoints) # OCC
    compute_snr_lm = np.zeros(b_adc_npoints) # Lloyd-Max
    compute_snr_cactus = np.zeros(b_adc_npoints) # CACTUS

    dp_unit = qr_bpbs_dp_unit(n_phy_rows=dp_dim, b_w=1, b_x=1, b_adc=7, sigma_adc=sigma_adc, c_qr_mean=1, v_dd=0.9)

    niter_cont = 1000

    for i in range(b_adc_npoints):
        dp_unit.update_b_adc(b_adc_options[i])
        compute_snr_fr[i] = dp_unit.compute_snr(npoints)
        print(f"Compute SNR with full range ADC having B_ADC = {b_adc_options[i]} : {compute_snr_fr[i]} dB")
        logging.info(f"Compute SNR with full range ADC having B_ADC = {b_adc_options[i]} : {compute_snr_fr[i]} dB")
    np.save(f'Outputs/compute_snr_fr_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.npy',compute_snr_fr)

    for i in range(b_adc_npoints):
        dp_unit.clip_adc_occ_gaussian(b_adc_options[i])
        compute_snr_occ[i] = dp_unit.compute_snr(npoints)
        print(f"Compute SNR with OCC-based ADC having B_ADC = {b_adc_options[i]} : {compute_snr_occ[i]} dB")
        logging.info(f"Compute SNR with OCC-based ADC having B_ADC = {b_adc_options[i]} : {compute_snr_occ[i]} dB")
    np.save(f'Outputs/compute_snr_occ_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.npy',compute_snr_occ)

    for i in range(b_adc_npoints):
        dp_unit.clip_adc_lloyd_max_gaussian(b_adc_options[i], niter_cont)
        compute_snr_lm[i] = dp_unit.compute_snr(npoints)
        print(f"Compute SNR with LM-based ADC having B_ADC = {b_adc_options[i]} : {compute_snr_lm[i]} dB")
        logging.info(f"Compute SNR with LM-based ADC having B_ADC = {b_adc_options[i]} : {compute_snr_lm[i]} dB")
    np.save(f'Outputs/compute_snr_lm_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.npy',compute_snr_lm)

    for i in range(b_adc_npoints):
        dp_unit.clip_adc_cactus(b_adc_options[i])
        compute_snr_cactus[i] = dp_unit.compute_snr(npoints)
        print(f"Compute SNR with CACTUS-based ADC having B_ADC = {b_adc_options[i]} : {compute_snr_cactus[i]} dB")
        logging.info(f"Compute SNR with CACTUS-based ADC having B_ADC = {b_adc_options[i]} : {compute_snr_cactus[i]} dB")
    np.save(f'Outputs/compute_snr_cactus_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.npy',compute_snr_cactus)


    plt.figure(figsize=(6,6))
    plt.plot(b_adc_options, compute_snr_fr, '-o',label='FR')
    plt.plot(b_adc_options, compute_snr_occ, '-o', label='OCC')
    plt.plot(b_adc_options, compute_snr_lm, '-o', label='LM')
    plt.plot(b_adc_options, compute_snr_cactus, '-o', label='CACTUS')
    plt.xlabel('ADC precision (bits)')
    plt.ylabel('Compute SNR (dB)')
    plt.xticks(b_adc_options)
    plt.legend()
    plt.grid()
    plt.savefig(f'Figures/imc_snr_b_adc_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.png', format='png', bbox_inches='tight')
    # plt.show()

def plot_results(dp_dim, sigma_adc,date):
    # Plots compute SNR vs ADC precision for all methods

    b_adc_options = np.array(range(3, 10))
    compute_snr_fr = np.load(f'Outputs/compute_snr_fr_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.npy')
    compute_snr_occ = np.load(f'Outputs/compute_snr_occ_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.npy')
    compute_snr_lm = np.load(f'Outputs/compute_snr_lm_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.npy')
    compute_snr_cactus = np.load(f'Outputs/compute_snr_cactus_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.npy')

    plt.figure(figsize=(6,4))
    plt.plot(b_adc_options, compute_snr_fr, '-o',label='FR')
    plt.plot(b_adc_options, compute_snr_occ, '-o', label='OCC')
    plt.plot(b_adc_options, compute_snr_lm, '-o', label='LM')
    plt.plot(b_adc_options, compute_snr_cactus, '-o', label='CACTUS (ours)')
    plt.xlabel('ADC precision (bits)')
    plt.ylabel('Compute SNR (dB)')
    plt.xticks(b_adc_options)
    plt.legend(fontsize = 9)
    plt.grid()
    plt.ylim((-3,53))
    plt.savefig(f'Figures/imc_snr_b_adc_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.png', format='png', bbox_inches='tight')


    # # === Annotations ===
    # # Horizontal arrow from x=7 to x=6 at y=compute_snr_occ
    # x_start = 8
    # x_end = 7
    # x_end_2 = 6
    # idx_5 = 3
    # idx_6 = 4
    # idx_7 = 5
    # y_occ_7 = compute_snr_occ[idx_7]
    # y_cactus_6 = compute_snr_cactus[idx_6]
    # y_cactus_5 = compute_snr_cactus[idx_5]

    # plt.annotate('', xy=(x_end_2-0.05, y_occ_7), xytext=(x_start, y_occ_7),
    #             arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # # Vertical arrow from y=compute_snr_occ to y=compute_snr_cactus at x=6
    # plt.annotate('', xy=(x_end_2, y_cactus_5), xytext=(x_end_2, y_occ_7),
    #             arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    # # plt.annotate('', xy=(x_end, 53), xytext=(x_end, y_occ_7),
    # #             arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    # # Label the CSNR gain
    # csnr_gain = y_cactus_5 - y_occ_7
    # plt.text(x_end_2 + 0.1, (y_occ_7 + y_cactus_5)/2, f'{csnr_gain:.1f} dB',
    #         va='center', fontsize=10)
    # # plt.text(x_end + 0.1, (y_occ_7 + 53)/2, '>20 dB',
    # #         va='center', fontsize=10)
    
    # # plt.annotate('', xy=(x_end_2-0.05, y_occ_7), xytext=(x_end, y_occ_7),
    # #             arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # # # Vertical arrow from y=compute_snr_occ to y=compute_snr_cactus at x=6
    # # plt.annotate('', xy=(x_end_2, y_cactus_5), xytext=(x_end_2, y_occ_7),
    # #             arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # # Label the CSNR gain
    # # csnr_gain = y_cactus_5 - y_occ_7 - 0.1
    # # plt.text(x_end_2 + 0.1, (y_occ_7 + y_cactus_5)/2, f'{csnr_gain:.1f} dB',
    #         # va='center', fontsize=10)

    # # Optional: Label bit saving on the horizontal arrow
    # # plt.text((x_end + x_end_2)/2, y_occ_7 + 1.5, '-1 bit', ha='center', fontsize=10)

    # plt.savefig(f'Figures/arrow_formatted_imc_snr_b_adc_N_{dp_dim}_sigma_adc_{sigma_adc}_date_{date}.png', format='png', bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    # Example usage

    run_simulation(dp_dim=128, sigma_adc=0.0005, npoints=500000, date= 325)
    run_simulation(dp_dim=128, sigma_adc=0.00075, npoints=500000, date= 325)
    run_simulation(dp_dim=128, sigma_adc=0.001, npoints=500000, date= 325)
    run_simulation(dp_dim=256, sigma_adc=0.0005, npoints=500000, date= 325)
    run_simulation(dp_dim=256, sigma_adc=0.00075, npoints=500000, date= 325)
    run_simulation(dp_dim=256, sigma_adc=0.001, npoints=500000, date= 325)
    run_simulation(dp_dim=512, sigma_adc=0.0005, npoints=500000, date= 325)
    run_simulation(dp_dim=512, sigma_adc=0.00075, npoints=500000, date= 325)
    run_simulation(dp_dim=512, sigma_adc=0.001, npoints=500000, date= 325)

    # plot_results(dp_dim=128, sigma_adc=0.0005, date= 325)
    # plot_results(dp_dim=128, sigma_adc=0.00075, date= 325)
    # plot_results(dp_dim=128, sigma_adc=0.001,date= 325)
    # plot_results(dp_dim=256, sigma_adc=0.0005, date= 325)
    # plot_results(dp_dim=256, sigma_adc=0.00075, date= 325)
    # plot_results(dp_dim=256, sigma_adc=0.001, date= 325)
    # plot_results(dp_dim=512, sigma_adc=0.0005, date= 325)
    # plot_results(dp_dim=512, sigma_adc=0.00075, date= 325)
    # plot_results(dp_dim=512, sigma_adc=0.001,date= 325)

    