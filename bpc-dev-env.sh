#!/bin/bash

# In terminal: bash bpc-dev-env.sh

# conda env create -f bpc-conda-env.yaml
# conda activate bpc

python -m venv bpc
source bpc/bin/activate

pip install -r bpc-requirements.txt
python -m ipykernel install --user --name=bpc

# Check out the latest Nilearn sources:
git clone https://github.com/nilearn/nilearn.git

# Navigate to your cloned Nilearn base directory and install:
pip install -e ./nilearn

deactivate

#echo "to remove enviroment run 'rm -rf bpc'"
#echo "'rm -rf /home/jupyter/.local/share/jupyter/kernels/bpc'"