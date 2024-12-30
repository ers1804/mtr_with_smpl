#!/bin/bash -l
#SBATCH --job-name=jepa_loss_trial
#SBATCH --output=/mnt/md0/erik/outputs/jepa_av2_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Activate Conda
#module add python
#module add cuda/12.3.0
#module add gcc/12.1.0
source /mnt/md0/erik/mtr_unitraj/bin/activate

#mkdir -p $TMPDIR/processed_scenarios_training
#mkdir -p $TMPDIR/processed_scenarios_validation
# find the data
#STORAGE_DIR="$(ws_find jepa_data)"
# the -P parameter defines the number of parallel processes, something like 4-8 should work well
#ls -1 $STORAGE_DIR/archives_val | xargs -P 8 -I{} tar xzf $STORAGE_DIR/archives_val/{} -C $TMPDIR
#ls -1 $STORAGE_DIR/archives_train | xargs -P 8 -I{} tar xzf $STORAGE_DIR/archives_train/{} -C $TMPDIR
#cp $WORK/processed_scenarios_training_infos.pkl $TMPDIR/processed_scenarios_training_infos.pkl
#cp $WORK/processed_scenarios_val_infos.pkl $TMPDIR/processed_scenarios_val_infos.pkl

# Unpack training data to $TMPDIR
#cd $TMPDIR
#tar xzf $WORK/mtr_training_wo_poses.tar.gz

set -x

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

cd /mnt/md0/erik/mtr_with_smpl/unitraj

export OMP_NUM_THREADS=8

python train.py method=MTR

# Deactivate the virtual environment at the end
deactivate