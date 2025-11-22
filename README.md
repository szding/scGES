# scGES
Integrating and mapping single-cell transcriptomics across the entire gene expression space

## Overview
Schematic view of the scGES framework.
![scGES_Overview](https://github.com/szding/scGES/blob/main/scGES_Overview.png)

**a**, The scGESI model integrates scRNA-seq data from different batches to construct a reference atlas and obtain harmonized and denoised data in the entire gene expression space. Different modules, scGESI-HVG and scGESI-LVG, are designed for HVGs and LVGs, respectively. **b**, The scGESM model projects new data onto the built reference atlas for harmonized and denoised gene expression. Using transfer learning, its two modules (scGESM-HVG and scGESM-LVG) leveraged network parameters learned from scGESI. Guided by scGESM-HVG’s predicted labels, scGESM-LVG aligns query data with the reference atlas for LVGs without label information.


## Installation
The scGES package is developed based on the Python libraries [Scanpy](https://scanpy.readthedocs.io/en/stable/) and [Pytorch](https://pytorch.org/https://pytorch.org/) framework, and can be run on GPU (recommend) or CPU.

First clone the repository. 

```
git clone https://github.com/szding/scGES.git
cd scGES-main
```

It's recommended to create a separate conda environment for running scGES:

```
#create an environment called scGES
conda create -n scGES python=3.9

#activate your environment
conda activate scGES
```
Install all the required packages. 

For Linux
```
pip install -r requirement.txt
```
Install scGES.

```
python setup.py build
python setup.py install
```

## Tutorials

Three step-by-step tutorials are included in the `Tutorial` folder to show how to use scGES.
- Tutorial_pancreas: Human pancreas data with nine batches.
- Tutorial_pbmc: Human PBMC data with nine batches.
- Tutorial_Lung: Human lung data with 16 batches.
- Tutorial_HSC: This data consists of 774 cells from SS2 platform and 2401 cells from MARS platform.
- Monocle_scTAE: The Monocle3 analysis of scGES's results.

## Support
If you have any questions, please feel free to contact us dszspur@xju.edu.cn. 
