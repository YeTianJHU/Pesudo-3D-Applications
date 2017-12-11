#!/bin/bash -l

#SBATCH
#SBATCH --job-name=8
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=end
#SBATCH --mail-user=ytian27@jhu.edu

#### load and unload modules you may
module load cuda/9.0
module load singularity/2.4
module load git

echo "Using GPU Device:"
echo $CUDA_VISIBLE_DEVICES

# python /home-4/ytian27@jhu.edu/scratch/yetian/C3D-TCN-Keras/VGG_Features.py --gpu=$CUDA_VISIBLE_DEVICES > /home-4/ytian27@jhu.edu/scratch/yetian/C3D-TCN-Keras//OUT-VGG.log

# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER 

singularity exec --nv /scratch/groups/singularity_images/pytorch.simg python train.py --machine=marcc --gpuid=$CUDA_VISIBLE_DEVICES --lr_steps 30 60 --save=8

# In the case of RuntimeError: cuda runtime error (48) : no kernel image is available for execution on the device at /tmp/pip-ds_7ifa8-build/aten/src/THCUNN/generic/Threshold.cu:34
# singularity pull --name pytorch.simg shub://marcc-hpc/pytorch
# singularity exec --nv ./pytorch.simg python train.py --machine=marcc --gpuid=$CUDA_VISIBLE_DEVICES --lr_steps 30 60 --save=8

echo "Finished with job $SLURM_JOBID"
