import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class uniform_adc_imc:
    # Class for representing uniform ADC in an IMC system.
    #
    # Parameters:
    # b_adc        : ADC precision (bits)
    # t1           : Lower clipping threshold
    # tM           : Upper clipping threshold
    # sigma_adc    : Standard deviation of ADC thermal noise
    # v_cl         : Set of possible input voltages
    # delta_imc    : Difference between adjacent input voltage levels
    # v_lsb        : LSB voltage step size
    # t_vec        : ADC threshold vector
    # r_vec        : Output representation vector


    def __init__(self, b_adc=3, t1=0.5, tM=6.5, sigma_adc=0, v_cl = np.arange(8)):
        # Instantiate uniform ADC with the necessary parameters
        self.b_adc = b_adc
        self.t1 = t1
        self.tM = tM
        self.sigma_adc = sigma_adc
        self.v_cl = v_cl
        self.delta_imc = (v_cl[1]-v_cl[0])
        self.M = 2**self.b_adc -1
        self.t_vec = np.linspace(self.t1, self.tM, self.M)
        self.v_lsb = (self.tM-self.t1)/(self.M-1)
        self.r_vec = np.zeros(self.M+1)
        self.r_vec[0] = (self.t1 - self.v_lsb/2 - v_cl[0])/self.delta_imc
        self.r_vec[1:] = (self.t_vec + self.v_lsb/2 - v_cl[0])/self.delta_imc
        

    def print_params(self):
        # Print ADC parameters to verify correct instantiation 
        print(f"Parameters for this ADC are:")
        print(f"  Resolution (B_ADC): {self.b_adc} bits")
        print(f"  Standard Deviation of ADC thermal noise (sigma_ADC): {self.sigma_adc} V")
        print(f"  LSB voltage (v_lsb): {self.v_lsb} V")
        print(f"  Threshold vector (t = [t1 ... tM]): {self.t_vec} V")
        print(f"  Representation vector (r = [r0 ... rM]): {self.r_vec}")
        print(f"  Set of possible input voltages (v_cl = [v0 ... vN]): {self.v_cl}")

    def convert(self, v_in):
        # Convert analog voltage v_in to the corresponding representation point based on the ADC transfer function
        v_in_noisy = v_in + np.random.normal(0, self.sigma_adc)
        if(v_in_noisy < self.t1): return self.r_vec[0]
        elif(v_in_noisy >= self.t_vec[self.M-1]): return self.r_vec[-1]
        else:
            for i in range(self.M+1):
                if(v_in_noisy>=self.t_vec[i-1] and v_in_noisy<self.t_vec[i]): return self.r_vec[i]
    
    def plot_adc(self, v_min, v_max, npoints = 1000):
        # Plot the ADC transfer function
        v_in = np.linspace(v_min, v_max, npoints)
        yq = np.zeros(npoints)
        for i in range(npoints):
            yq[i] = self.convert(v_in[i])
        plt.figure(figsize=(6, 4))
        plt.plot(v_in, yq, label='ADC transfer function')
        for i in range(self.M):
            if(i==0):
                plt.axvline(x=self.t_vec[i], linestyle='--', color = 'red', linewidth=1, label = 'ADC thresholds')

            else:
                plt.axvline(x=self.t_vec[i], linestyle='--', color = 'red', linewidth=1)
        plt.plot(self.v_cl, np.arange(len(self.v_cl)), label='Ideal ADC transfer function')
        plt.scatter(self.v_cl, np.zeros(len(self.v_cl)), color = 'green', linewidth = 1, label = 'Ideal voltage-domain DP')
        plt.legend()
        plt.grid()
        plt.title(f' delta_imc = {self.delta_imc:.4f} V, sigma_ADC = {self.sigma_adc} V, B_ADC = {self.b_adc}', fontsize = 11)
        plt.xlabel("Analog input (V)")
        plt.ylabel("Digital output representation")
        plt.tight_layout()
        plt.show()

class non_uniform_adc_imc:
    # Class for representing non-uniform ADC in an IMC system.
    #
    # Parameters:
    # b_adc        : ADC precision (bits)
    # sigma_adc    : Standard deviation of ADC thermal noise
    # t_vec        : ADC threshold vector
    # r_vec        : Output representation vector
    # v_cl         : Set of possible input voltages

    def __init__(self, b_adc, t_vec, r_vec, sigma_adc=0, v_cl = np.arange(8)):
    # Instantiate uniform ADC with the necessary parameters
        self.b_adc = b_adc
        self.sigma_adc = sigma_adc
        self.M = 2**self.b_adc -1
        self.t_vec = t_vec
        self.r_vec = r_vec
        self.v_cl = v_cl

    def print_params(self):
    # Print ADC parameters to verify correct instantiation 
        print(f"Parameters for this ADC are:")
        print(f"  Resolution (B_ADC): {self.b_adc} bits")
        print(f"  Standard Deviation of ADC thermal noise (sigma_ADC): {self.sigma_adc} V")
        print(f"  Threshold vector (t = [t1 ... tM]): {self.t_vec}")
        print(f"  Representation vector (r = [r0 ... rM]): {self.r_vec}")
        print(f"  Set of possible input voltages (v_cl = [v0 ... vN]): {self.v_cl}")

    def convert(self, v_in):
    # Convert analog voltage v_in to the corresponding representation point based on the ADC transfer function
        v_in_noisy = v_in + np.random.normal(0, self.sigma_adc)
        if(v_in_noisy < self.t_vec[0]): return self.r_vec[0]
        elif(v_in_noisy >= self.t_vec[self.M-1]): return self.r_vec[-1]
        else:
            for i in range(self.M+1):
                if(v_in_noisy>=self.t_vec[i-1] and v_in_noisy<self.t_vec[i]): return self.r_vec[i]
    
    def plot_adc(self,  v_min, v_max, npoints = 1000):
    # Plot the ADC transfer function
        v_in = np.linspace(v_min, v_max, npoints)
        yq = np.zeros(npoints)
        for i in range(npoints):
            yq[i] = self.convert(v_in[i])
        plt.figure(figsize=(6, 4))
        plt.plot(v_in, yq, label='ADC transfer function')
        for i in range(self.M):
            if(i==0):
                plt.axvline(x=self.t_vec[i], linestyle='--', color = 'red', linewidth=1, label = 'ADC thresholds')
            else:
                plt.axvline(x=self.t_vec[i], linestyle='--', color = 'red', linewidth=1)
        plt.plot(self.v_cl, np.arange(len(self.v_cl)), label='Ideal ADC transfer function')
        plt.scatter(self.v_cl, np.zeros(len(self.v_cl)), color = 'green', linewidth = 1, label = 'Ideal voltage-domain DP')
        plt.xlabel("Analog input (V)")
        plt.ylabel("Digital output representation")
        plt.legend()
        plt.grid()
        plt.title(f' sigma_ADC = {self.sigma_adc} V, B_ADC = {self.b_adc}', fontsize = 11)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Example usage

    v_dd = 0.7
    N = 8
    delta_imc = v_dd/(1.3*N)
    v_cl = np.arange(0,N+1)*delta_imc

    # Uniform ADC test
    uni_adc = uniform_adc_imc(b_adc=3, t1 = 0.5*delta_imc, tM = 6.5*delta_imc, sigma_adc=0.0005, v_cl=v_cl)
    uni_adc.print_params()
    uni_adc.plot_adc(v_min = 0, v_max = v_dd, npoints = 5000)

    # Non-uniform ADC test with arbitrary t_vec and r_vec
    t_vec = np.array([0.1*v_dd, 0.3*v_dd, 0.4*v_dd, 0.45*v_dd, 0.5*v_dd, 0.6*v_dd,0.8*v_dd])
    r_vec = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    non_uni_adc = non_uniform_adc_imc(b_adc=3, t_vec=t_vec, r_vec=r_vec, sigma_adc=0.0005, v_cl=v_cl)
    non_uni_adc.print_params()
    non_uni_adc.plot_adc(v_min = 0, v_max = v_dd, npoints = 5000)