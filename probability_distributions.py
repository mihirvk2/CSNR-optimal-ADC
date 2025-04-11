import numpy as np
import scipy as scp

def binomial_pmf(N, p):
    # Computes the Binomial Probability Mass Function (PMF) for N trials and success probability p,
    # and calculates the mean and variance of the distribution.
    pmf = np.zeros(N+1)
    second_moment = 0
    mean = 0
    for i in range(N+1):
        pmf[i] = scp.special.comb(N,i)*(p**i)*(1-p)**(N-i)
        second_moment += pmf[i]*i**2
        mean += pmf[i]*i
    var = second_moment - mean**2
    return pmf, mean, var

def poisson_pmf(N,lam):
    # Computes the Poisson Probability Mass Function (PMF) for a given rate lam,
    # and calculates the mean and variance of the distribution.
    pmf = np.zeros(N+1)
    second_moment = 0
    mean = 0
    for i in range(N+1):
        pmf[i] = (lam**i) * np.exp(-lam) / scp.factorial(i)
        second_moment += pmf[i] * i**2
        mean += pmf[i] * i
    var = second_moment - mean**2
    return pmf, mean, var

def gaussian_pdf(x, mu, sigma):
    # Computes the Probability Density Function (PDF) of a Gaussian (Normal) distribution
    # with mean mu and standard deviation sigma.
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*((x-mu)/sigma)**2)

def x_times_p_x_gaussian(x, N, p, delta_imc):
    # Computes the value of x * p(x) for a Gaussian distribution with mean N*p*delta_imc
    # and standard deviation sqrt(N*p*(1-p))*delta_imc.
    return x*gaussian_pdf(x, N*p*delta_imc, np.sqrt(N*p*(1-p))*delta_imc)

def cl_voltage_pdf(x, N, p, sigma_ADC, alpha):
    # Computes the PDF of the capacitance line voltage in IMC, assuming a binomial distribution
    # of ideal dot product outputs    
    pmf, sig_mean, sig_var = binomial_pmf(N, p)
    # print(pmf)
    sum = 0
    for i in range(N+1):
        # print(gaussian_pdf(x, alpha*i,sigma_ADC))
        sum+= pmf[i]*gaussian_pdf(x, alpha*i,sigma_ADC)
    return sum

def normal_cdf(x):
    # Computes the CDF of the standard normal distribution using the error function (erf).
    return 0.5*(1+scp.special.erf(x/np.sqrt(2)))