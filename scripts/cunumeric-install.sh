#!/usr/bin/env bash
# Override branches by setting environment variables when invoking the script.
# Example:
#   LEGATE_BRANCH=main CUNUMERIC_BRANCH=main curl -fsSL https://krasow.dev/scripts/cunumeric-install.sh | bash -s --
# Defaults to 'develop' when not provided.

LEGATE_BRANCH=${LEGATE_BRANCH:-develop}
CUNUMERIC_BRANCH=${CUNUMERIC_BRANCH:-develop}

cd $HOME
curl -fsSL https://install.julialang.org | bash -s -- --default-channel 1.11 --yes

source ~/.bashrc

# Install CMAKE v3.30
cd $HOME && \
    wget https://github.com/Kitware/CMake/releases/download/v3.30.7/cmake-3.30.7-linux-x86_64.sh --no-check-certificate && \
    sh cmake-3.30.7-linux-x86_64.sh --skip-license --prefix=$HOME/.local

cd $HOME && \
    git clone -b $LEGATE_BRANCH https://github.com/JuliaLegate/Legate.jl && \
    git clone -b $CUNUMERIC_BRANCH https://github.com/JuliaLegate/cuNumeric.jl


cd $HOME/cuNumeric.jl
julia --project=. -e 'using Pkg; Pkg.develop("../"); Pkg.develop("../lib/LegatePreferences"); Pkg.develop("./lib/CNPreferences")'
julia --project=. -e 'using LegatePreferences; LegatePreferences.use_developer_mode(); using CNPreferences; CNPreferences.use_developer_mode();'
julia --project=. -e 'using Pkg; Pkg.build("cuNumeric")'

# conda install for cupynumeric
mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh --no-check-certificate && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh && \
    source ~/miniconda3/bin/activate

# install cupynumeric
conda init bash && \
    source ~/.bashrc && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    bash $HOME/cuNumeric.jl/benchmark/install_cupynumeric.sh

