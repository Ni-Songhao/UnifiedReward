export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1


export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2100

NNODES=$HOST_NUM
echo "NNODES: $NNODES"
NPROC_PER_NODE=$HOST_GPU_NUM
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
NODE_RANK=$INDEX
echo "NODE_RANK: $NODE_RANK"
DATA_NUM_WORKERS=4
echo "DATA_NUM_WORKERS: $DATA_NUM_WORKERS"
SP_SIZE=2

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
BASE_RUN_NAME="UnifiedReward-Think-GRPO"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

PROMPT_VERSION="qwen_1_5"
RUN_NAME="UnifiedReward-Think-GRPO" 
PREV_STAGE_CHECKPOINT="your/rejection_sampling_model/path"
DATA_PATH="dataset/grpo.yaml"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

hostfile="your/hostfile"

DS_SKIP_CUDA_CHECK=1 deepspeed --hostfile "$hostfile" --master_addr "${LOCAL_IP}" --master_port 29502 \
    llava/train/train_grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --ddp_timeout 180000000 \
    --output_dir ./checkpoints/$RUN_NAME  \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --data_path $DATA_PATH \
    --dataloader_num_workers $DATA_NUM_WORKERS \
    --dataset_name xxx \
    --max_prompt_length 2048 \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --beta 0.04 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_total_limit 20