job_name="vggt_1d_uncomplete_8datasets"
base_logdir="/lustre/fsw/portfolios/nvr/users/ymingli/projects/VideoActionModel/3DV/vggt/training/logs"

for i in {1..8}; do
    submit_job --gpu 8 --cpu 24 --nodes 1 --partition=grizzly,polar,polar3,polar4 --account=nvr_av_end2endav \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/cu118.sqsh  \
    --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
    --duration 4 \
    --dependency=singleton \
    --name $job_name \
    --logdir ${base_logdir}/${job_name}/run_${i} \
    --notimestamp \
    --command  "bash run.sh"
done