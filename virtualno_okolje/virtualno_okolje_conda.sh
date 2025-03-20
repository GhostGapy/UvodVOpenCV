#!/bin/bash

# Preveri, če je conda nameščena
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Installing Miniconda..."
    
    # Namesti condo
    # Ustrezni paket najdete na https://docs.conda.io/en/latest/miniconda.html
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    rm miniconda.sh
fi

# Kje se nahaja opis okolja za condo
ENV_FILE="okolje_conda.yml"

# Ustvari okolja na podlagi opisa v environment.yml
conda env create -f $ENV_FILE

# Aktiviraj okolje
conda activate my_environment

# Katere pakete imamo nameščene
conda list

# Deaktiviraj okolje
#conda deactivate
