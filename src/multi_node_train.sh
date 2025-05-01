#!/bin/bash

# Configuration
MAIN_HOST=192.168.1.10  # Master node IP
MAIN_PORT=29500         # Communication port
NUM_MACHINES=3
NUM_PROCESSES=16        # 8 + 4 + 4
CONFIG_PATH="$HOME/.cache/huggingface/accelerate/default_config.yaml"
TRAIN_SCRIPT="/home/user/project/src/model.py"

# Node-specific ranks
MACHINES=(
    "192.168.1.10:0:8"
    "192.168.1.11:1:4"
    "192.168.1.12:2:4"
)

# Launch function
launch_node() {
    IP=$1
    RANK=$2
    SLOTS=$3
    CMD="accelerate launch --multi_gpu --num_processes=$NUM_PROCESSES --num_machines=$NUM_MACHINES --machine_rank=$RANK \
         --main_process_ip=$MAIN_HOST --main_process_port=$MAIN_PORT \
         --config_file $CONFIG_PATH $TRAIN_SCRIPT"

    if [[ "$RANK" == "0" ]]; then
        echo "Launching training on master node (rank 0)..."
        eval "$CMD"
    else
        echo "Launching training on node $IP (rank $RANK)..."
        ssh $IP "$CMD" &
    fi
}

# Loop over machine configs and launch
for entry in "${MACHINES[@]}"; do
    IFS=":" read -r IP RANK SLOTS <<< "$entry"
    launch_node "$IP" "$RANK" "$SLOTS"
done

wait
echo "Distributed training launched across all nodes."
