#!/bin/bash

# Configuration
MAIN_HOST=10.22.14.246  # Master node IP
MAIN_PORT=29599       # Changed port to avoid conflicts
NUM_MACHINES=2          # Total number of machines
NUM_PROCESSES=8         # REDUCED: Start with fewer processes per machine
CONFIG_PATH="$HOME/.cache/huggingface/accelerate/default_config.yaml"
TRAIN_SCRIPT="/home/sureshm/ssuresh/ACCfiy/src/train.py"
CONDA_ENV="llm"
TIMEOUT=1800           # Increased timeout to 30 minutes

# Environment variables to debug the distributed setup
# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Enable detailed debugging
# export TORCH_CPP_LOG_LEVEL=INFO        # More C++ logs
# export NCCL_DEBUG=INFO                 # NCCL debugging
# export CUDA_LAUNCH_BLOCKING=1          # Synchronous CUDA

# Free up GPUs by killing any zombie processes
kill_gpus() {
    echo "Freeing up GPU resources..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read -r pid; do
        echo "Killing process $pid"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3  # Wait for resources to free up
    echo "GPU resources should now be available"
}

# Check connectivity between nodes
check_network() {
    local target_host=$1
    echo "Testing connectivity to $target_host..."
    
    # Basic ping test
    ping -c 3 $target_host || echo "Warning: Cannot ping $target_host"
    
    # Port availability check
    if [[ "$(hostname -I)" =~ $MAIN_HOST ]]; then
        # On main node, check if port is available
        if netstat -tuln | grep -q ":$MAIN_PORT "; then
            echo "Warning: Port $MAIN_PORT is already in use"
            echo "Killing processes using port $MAIN_PORT..."
            fuser -k $MAIN_PORT/tcp 2>/dev/null || true
            sleep 2
        fi
        
        # Open firewall for the port
        sudo ufw allow $MAIN_PORT/tcp 2>/dev/null || echo "Could not configure firewall"
        
        # Start a temporary listener to test
        echo "Testing port with temporary listener..."
        (nc -l $MAIN_PORT > /dev/null 2>&1) &
        NC_PID=$!
        sleep 2
        kill $NC_PID 2>/dev/null
    else
        # On worker nodes, try to connect to the main node port
        echo "Testing connection to $MAIN_HOST:$MAIN_PORT..."
        nc -z -w 5 $MAIN_HOST $MAIN_PORT || echo "Warning: Cannot connect to $MAIN_HOST:$MAIN_PORT"
    fi
}

# Launch function
launch_node() {
    IP=$1
    RANK=$2
    SLOTS=$3
    
    # Command to launch training with explicit environment variables
    CMD="TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 \
         accelerate launch \
         --multi_gpu \
         --num_processes=$NUM_PROCESSES \
         --num_machines=$NUM_MACHINES \
         --machine_rank=$RANK \
         --main_process_ip=$MAIN_HOST \
         --main_process_port=$MAIN_PORT \
         --config_file $CONFIG_PATH $TRAIN_SCRIPT"

    if [[ "$(hostname -I)" =~ $IP ]]; then
        # We're on this node, run locally
        echo "Launching training on local node (rank $RANK)..."
        kill_gpus
        eval "$CMD"
    else
        echo "Launching training on remote node $IP (rank $RANK)..."
        # Test SSH connection first
        ssh -o ConnectTimeout=5 $IP "echo Connection to $IP successful" || {
            echo "ERROR: Cannot connect to $IP via SSH"
            return 1
        }
        
        # Kill any processes using GPUs on the remote node
        ssh $IP "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 3"
        
        # Launch on remote node with detailed debugging
        ssh $IP "source /home/sureshm/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $CMD" &
    fi
}

# Main execution
echo "Starting distributed training setup..."

# Only use the first two GPUs for simpler debugging
export CUDA_VISIBLE_DEVICES=0,1


# Run on current node
if [[ "$(hostname -I)" =~ $MAIN_HOST ]]; then
    # We're on the main node
    echo "Running setup on main node: $MAIN_HOST"
    kill_gpus
    check_network $MAIN_HOST
else
    # We're on a worker node
    echo "Running setup on worker node: $(hostname -I)"
    kill_gpus
    check_network $MAIN_HOST
fi

# Node-specific ranks
MACHINES=(
    "10.22.14.246:0:4"  # main node uses 2 GPUs
    "10.22.14.245:1:4"  # worker node uses 2 GPUs
)

# First launch rank 0 (master) and wait for it to start
for entry in "${MACHINES[@]}"; do
    IFS=":" read -r IP RANK SLOTS <<< "$entry"
    if [[ "$RANK" == "0" ]]; then
        echo "Launching master node (rank 0)..."
        launch_node "$IP" "$RANK" "$SLOTS"
        echo "Waiting for master node to start..."
        sleep 30  # Give master node time to initialize
        break
    fi
done


wait
echo "All training processes have completed."

# Restore the original config
if [ -f /home/sureshm/ssuresh/ACCfiy/src/config.yaml.bak ]; then
    echo "Restoring original configuration..."
    mv /home/sureshm/ssuresh/ACCfiy/src/config.yaml.bak /home/sureshm/ssuresh/ACCfiy/src/config.yaml
fi