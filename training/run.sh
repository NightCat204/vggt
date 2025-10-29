source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate vggt

# Block default docker packages (refer to common issues)
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu

# Configure your WandB account
bash /lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/squash_data/co3d/script/unmount_squash.sh
bash /lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/squash_data/DL3DV/script/unmount_squash.sh
bash /lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/squash_data/WildRGB-D/script/unmount_squash.sh
bash /lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/squash_data/Omnidata/script/unmount_squash.sh

bash /lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/squash_data/co3d/script/mount_squash.sh
bash /lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/squash_data/DL3DV/script/mount_squash.sh
bash /lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/squash_data/WildRGB-D/script/mount_squash.sh
# bash /lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/squash_data/Omnidata/script/mount_squash.sh

export WANDB_API_KEY=33c29c845cd7aef05a73bed11c05caa4a2e8d120
torchrun --standalone --nnodes=1 --nproc_per_node=8 launch.py
# torchrun --standalone --nnodes=1 --nproc_per_node=1 launch.py --config default_1d_co3d