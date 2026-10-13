#!/bin/bash

# Stop the script immediately if any command fails
set -e 

GPU_ID=${1:-0}
#EXPLAINERS=${2:-"random tam inputxgradients integratedgradients rollout gradxrollout gradcam llavacam lxt"}
EXPLAINERS=${2:-"tam"}
MODEL_NAME=${3:-"llava_7b"}
DATASET_NAME=${4:-"gridvqa"}


echo "========================================="
echo "   Starting VLM XAI Benchmark Suite      "
echo "========================================="


# 1. Define your configurations
# Add or remove paths here to change what the script runs
MODELS="configs/models/${MODEL_NAME}.yaml"

if [ ! -f "$MODELS" ]; then
    echo "ERROR : The file $MODEL_CONFIG does not exist !"
    exit 1
fi

# DATASETS="configs/datasets/${DATASET_NAME}.yaml"

# if [ ! -f "$DATASETS" ]; then
#     echo "ERROR : The file $MODEL_CONFIG does not exist !"
#     exit 1
# fi

OUTPUT_DIR=(
    "logs/gridvqa"
)



# 2. Nested loops to run every combination
for MODEL in "${MODELS[@]}"; do
        
    echo ""
    echo ">>> Launching Job: Model=$MODEL | Dataset=$DATASET_NAME"
    echo ">>> Explainers: $EXPLAINERS"
    echo ">>> Executing module: src.benchmarks.$DATASET_NAME"
    
    # 3. Execute the Python script
    python -m src.benchmarks."${DATASET_NAME}" \
        --model_config "$MODEL" \
        --dataset_name "$DATASET_NAME" \
        --gpu_id $GPU_ID \
        --output_dir $OUTPUT_DIR \
        --explainers $EXPLAINERS \
        #--max_samples 2
        
    echo "<<< Job Finished. Moving to next configuration..."
    sleep 5 # Give the OS a few seconds to fully flush the GPU VRAM
        
done

echo "========================================="
echo "   All Benchmarks Completed Successfully! "
echo "========================================="
