# bpc_jupyter

This Jupyter notebook is a tutorial for how to calculate Basis Profile Curve for intracranial EEG data during single pulse stimulation as described in: 

Miller KJ, Müller K-R, Hermes D (2021) Basis profile curve identification to understand electrical stimulation effects in human brain networks. PLoS Comput Biol 17(9): e1008710. https://doi.org/10.1371/journal.pcbi.1008710

Please cite this work when using the code. 

This Jupyter notebook is a translation of the Matlab code that was developed by Kai Miller in the paper. The Jupyter notebook working with MEF3 data was written by Tal Pal Attia, Harvey Huang and Dora Hermes, 2021. The Jupyter notebook working with Brainvision data was written by Alex Rockhill and Dora Hermes, 2023. 

## Data preparations
Download this example dataset from OpenNeuro: https://openneuro.org/datasets/ds003708

## Python packages
### Working with raw MEF3 data (out of date)
`bpc_interactive.ipynb`
Set up the environment and add the required python packages by opening the jupiter notebook (bpc_interactive.ipynb) and completing **Section 1, Getting started with the Python environment and packages** 

### Working with preprocessed Brainvision data
`bpc_interactive_mne.ipynb` 
Data in the /derivatives/preprocessed repository have been referenced to an adjusted common average. This simplifies the code and process, and you can use mne python to work with these Brainvision data. Code depends on the following Python packages:
- MNE
- MNE-BIDS
- openneuro-py
- numpy (version>=1.24.4)
- pandas (version>=2.0.3)
- scipy (version>=1.10.1)
- sklearn
- matplotlib
- tqdm
- ipykernel
- nilearn



## Acknowledgements
This project was funded by the National Institute Of Mental Health of the National Institutes of Health under Award Number R01MH122258.


