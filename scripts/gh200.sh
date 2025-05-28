#!/bin/bash

set -ex

conda init bash
conda activate vllm_env

export CODE_HOME=/home/sureshm/ACCfiy/
# Launch command with improved arguments
accelerate launch \
    --config_file "$CODE_HOME/src/run_config/accel/qwen_accel.yaml" \
    --num_processes 1 \
    --num_machines 1 \
    --main_process_ip 10.22.12.187 \
    --main_process_port 29502 \
    --mixed_precision "fp16" \
    --use_deepspeed \
    --deepspeed_config_file "$CODE_HOME/src/run_config/deepspd/qwen_ds.json" \
    --gradient_accumulation_steps 2 \
    --zero3_init_flag True \
    --dynamo_backend=no \
    "$CODE_HOME/main.py"