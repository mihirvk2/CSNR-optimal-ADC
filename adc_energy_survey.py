import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Data

    sram_sorted_refs = ["JSSC'20 [5]", "JSSC'20 [6]", "CICC'21 [7]", "VLSI'21 [8]", "CICC'22 [9]", "JSSC'23 [10]", "JSSC'23 [11]", "ESSERC'24 [12]", "ESSERC'24 [13]", "JSSC'24 [14]", "JSSC'24 [15]"]
    sram_sorted_values = np.array([0.387, 0.156, 0.21, 0.588, 0.497, 0.18, 0.34, 0.612, 0.3943, 0.197561, 0.5])*100
 
    envm_sorted_refs = ["VLSI'22 [16]", "SSCL'24 [17]", "ESSERC'24 [18]"]
    envm_sorted_values = np.array([0.271, 0.59, 0.318])*100

    plt.figure(figsize=(6, 4))
    # Bar plots with red color but distinguishable using hatching
    plt.bar(range(len(sram_sorted_refs)), sram_sorted_values, 
        color='blue',label='SRAM-based A-IMC')
    plt.bar(range(len(sram_sorted_refs), len(sram_sorted_refs) + len(envm_sorted_refs)), 
            envm_sorted_values, color='green', label='eNVM-based A-IMC')
    # X-ticks and labels
    plt.xticks(range(len(sram_sorted_refs) + len(envm_sorted_refs)), 
            sram_sorted_refs + envm_sorted_refs, rotation=70, fontsize=10)
    plt.yticks(fontsize=11)
    plt.ylabel("ADC energy ($\%$)", fontsize=14)
    # Horizontal reference line
    plt.axhline(y=100, color='black', linewidth=1.5)
    plt.text(6.5, 103, "Total A-IMC energy", ha='center', fontsize=13, color='black')
    plt.ylim((0, 115))
    plt.tight_layout()
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
    plt.grid()
    plt.savefig('Figures/adc_energy_contribution.png', format='png', bbox_inches='tight')
    plt.show()