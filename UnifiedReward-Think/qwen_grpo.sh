#!/bin/bash
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
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

export NCCL_P2P_DISABLE=1

export NCCL_P2P_LEVEL=NVL         
export NCCL_LAUNCH_MODE=GROUP


export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4300

NNODES=$HOST_NUM
echo "NNODES: $NNODES"
NPROC_PER_NODE=$HOST_GPU_NUM
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
NODE_RANK=$INDEX
echo "NODE_RANK: $NODE_RANK"
DATA_NUM_WORKERS=4
echo "DATA_NUM_WORKERS: $DATA_NUM_WORKERS"
SP_SIZE=2

RUN_NAME="UnifiedReward-Think-qwen-GRPO" 
export DATA_PATH=dataset/grpo.yaml
export CKPT_PATH="your/rejection_sampling_model/path"
export SAVE_PATH=./checkpoints/$RUN_NAME

hostfile="/path/to/hostfile"

DS_SKIP_CUDA_CHECK=1 deepspeed --hostfile "$hostfile" --master_addr "${LOCAL_IP}" --master_port 29502 \
    src/open_r1/grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --ddp_timeout 180000000 \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 8192 \
    --max_completion_length 4096 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 147456 \
    --save_steps 40 \
    --save_total_limit 8 \
    --save_only_model false \
    --num_train_epochs 2