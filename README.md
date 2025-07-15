
This repository contains the source code for our paper [Compute SNR-Optimal Analog-to-Digital Converters for Analog In-Memory Computing](https://arxiv.org/abs/2507.09776). 

## Abstract
Analog in-memory computing (AIMC) is an energy-efficient alternative to digital architectures for accelerating machine learning and signal processing workloads. However, its energy efficiency is limited by the high energy cost of the column analog-to-digital converters (ADCs). Reducing the ADC precision is an effective approach to lowering its energy cost. However, doing so also reduces the AIMC's computational accuracy thereby making it critical to identify the minimum precision required to meet a target accuracy. Prior works overestimate the ADC precision requirements by modeling quantization error as input-independent noise, maximizing the signal-to-quantization-noise ratio (SQNR), and ignoring the discrete nature of ideal pre-ADC signal. We address these limitations by developing analytical expressions for estimating the compute signal-to-noise ratio (CSNR), a true metric of accuracy for AIMCs, and propose CACTUS, an algorithm to obtain CSNR-optimal ADC parameters. Using a circuit-aware behavioral model of an SRAM-based AIMC in a 28nm CMOS process, we show that for a 256-dimensional binary dot product, CACTUS reduces the ADC precision requirements by 3b while achieving 6dB higher CSNR over prior methods. We also delineate operating conditions under which our proposed CSNR-optimal ADCs outperform conventional SQNR-optimal ADCs.

## Citation
If you use any part of this work, please cite our paper:

```bibtex
@misc{kavishwar2025csnr,
      title={Compute SNR-Optimal Analog-to-Digital Converters for Analog In-Memory Computing}, 
      author={Mihir Kavishwar and Naresh Shanbhag},
      year={2025},
      eprint={2507.09776},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2507.09776}, 
}
```
