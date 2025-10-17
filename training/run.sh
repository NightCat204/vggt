source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate vggt

# Block default docker packages (refer to common issues)
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu

# Configure your WandB account
export WANDB_API_KEY=33c29c845cd7aef05a73bed11c05caa4a2e8d120
torchrun --standalone --nnodes=1 --nproc_per_node=1 launch.py