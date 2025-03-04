#export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=4
current_time=$(date +%Y%m%d%H%M%S)
#current_time=20240722100141

EXPNAME="vc2_turbo_dpo"
CONFIG='configs/t2v_turbo_dpo/config.yaml' # experiment config 
LOGDIR="./results/baselines"  # experiment saving directory all should under subfolder so that won't be copied to codeversion

python -m torch.distributed.run \
--nnodes=1 \
--nproc_per_node=4 \
--master_port=29500 \
scripts/train.py \
-t --devices '0,1,2,3' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "checkpoint_turbo_dpo" \
--logdir $LOGDIR \
--auto_resume True 
