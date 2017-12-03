#!/bin/bash -l

#SBATCH
#SBATCH --job-name=ucf
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
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

singularity exec --nv /scratch/groups/singularity_images/pytorch.simg python train.py --machine=ye_home --gpu=$CUDA_VISIBLE_DEVICES > output.log
echo "Finished with job $SLURM_JOBID"
