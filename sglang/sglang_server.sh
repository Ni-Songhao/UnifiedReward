python -m sglang.launch_server \
    --model-path CodeGoat24/UnifiedReward-7b-v1.5\
    --api-key unifiedreward\
    --port 8080\
    --chat-template chatml-llava\
    --enable-p2p-check\
    --mem-fraction-static 0.85
