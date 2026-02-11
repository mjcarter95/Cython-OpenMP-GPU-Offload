#!/bin/bash -l
#SBATCH -J cython_offload
#SBATCH -p gpu-a100-lowbig,gpu-a100-dacdt
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -o %j.out
#SBATCH --export=ALL
 
module purge
module load miniforge3/25.3.0-python3.12.10
module load cuda/12.8.0-gcc14.2.0
module load llvm/19.1.3-gcc14.2.0
module load llvm-openmp/18.1.0-gcc14.2.0
source activate /users/mcarter/fastscratch/cythongpu/env

which python3
python3 -m pip list

cd /users/mcarter/fastscratch/cythongpu

export OMP_TARGET_OFFLOAD=MANDATORY
export LIBOMPTARGET_INFO=4
export LIBOMPTARGET_DEBUG=1
export CUDA_VISIBLE_DEVICES=0

 
echo "=============================================================="
echo " Job running on node(s): $SLURM_NODELIST"
echo " GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "--------------------------------------------------------------"
echo " Compiler: $(clang++ --version | head -n 1)"
echo " CUDA version: $(nvcc --version | grep release)"
echo " LLVM libdir: $LLVM_LIBDIR"
echo " LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo " Start time: $(date)"
echo "=============================================================="
echo "clearing old files"
rm ./src/gpuomp/gpu.cpp
rm ./src/gpuomp/*.so
echo "=============================================================="
echo "build and install"
python3 -m pip install -e . --force-reinstall
echo "=============================================================="
python3 main.py
echo "=============================================================="
echo " End time: $(date)"
echo "=============================================================="
 
 