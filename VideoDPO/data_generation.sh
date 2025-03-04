python scripts/turbo_inference/text2video.py \
        --unet_dir "checkpoints/t2v-turbo/unet_lora.pt" \
        --base_model_dir checkpoints/vc2/model.ckpt \
        --prompts_file /path/to/prompts_file \
        --output_dir "./turbo_dpo_dataset"