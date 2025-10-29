submit_job --gpu 1 --cpu 16 --nodes 1 \
    --partition=grizzly,polar,polar3,polar4 \
    --account=nvr_av_end2endav \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/cu118.sqsh \
    --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
    --duration 4 \
    --dependency=singleton \
    --name vggt_1d_uncomplete_test \
    --logdir /lustre/fsw/portfolios/nvr/users/ymingli/projects/VideoActionModel/3DV/vggt/training/logs \
    --command "bash run_test.sh"