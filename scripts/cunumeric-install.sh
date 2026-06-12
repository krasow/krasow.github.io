#!/usr/bin/env bash
# Override branches by setting environment variables when invoking the script.
# Example:
# curl -fsSL https://krasow.dev/scripts/cunumeric-install.sh | LEGATE_BRANCH=develop CUNUMERIC_BRANCH=benchmark-fusion-again bash
# Defaults to 'develop' when not provided.

LEGATE_BRANCH=${LEGATE_BRANCH:-develop}
CUNUMERIC_BRANCH=${CUNUMERIC_BRANCH:-develop}

echo "Legate branch set to: ${LEGATE_BRANCH}"
echo "cuNumeric branch set to ${CUNUMERIC_BRANCH}"

cd $HOME
curl -fsSL https://install.julialang.org | bash -s -- --default-channel 1.11 --yes

# ~/.bashrc returns early non-interactively, so set PATH directly.
export PATH="$HOME/.juliaup/bin:$HOME/.local/bin:$PATH"

# Install CMAKE v3.30
# The CMake self-extractor cd's into --prefix, so the directory must exist first.
cd $HOME && \
    mkdir -p "$HOME/.local" && \
    wget https://github.com/Kitware/CMake/releases/download/v3.30.7/cmake-3.30.7-linux-x86_64.sh --no-check-certificate && \
    sh cmake-3.30.7-linux-x86_64.sh --skip-license --prefix=$HOME/.local

cd $HOME && \
    git clone -b $LEGATE_BRANCH https://github.com/JuliaLegate/Legate.jl && \
    git clone -b $CUNUMERIC_BRANCH https://github.com/JuliaLegate/cuNumeric.jl


cd $HOME/cuNumeric.jl
julia --project=. -e 'using Pkg; Pkg.develop(path="../Legate.jl"); Pkg.develop(path="../Legate.jl/lib/LegatePreferences"); Pkg.develop(path="./lib/CNPreferences")'
julia --project=. -e 'using LegatePreferences; LegatePreferences.use_developer_mode(); using CNPreferences; CNPreferences.use_developer_mode();'
julia --project=. -e 'using Pkg; Pkg.build("cuNumeric")'

cd $HOME/cuNumeric.jl/benchmark
julia --project=. -e 'using Pkg; Pkg.develop(path="../"); Pkg.resolve(); Pkg.instantiate();'

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

