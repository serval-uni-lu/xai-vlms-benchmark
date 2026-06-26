#!/bin/bash

# Stop the script immediately if any command fails
set -e 

GPU_ID=${1:-0}
MODEL_NAME=${2:-"internvl_2b"}
DATASET_NAME=${3:-"mmstar"}
EXPLAINERS=${4:-"random gradcam tam llavacam inputxgradients integratedgradients rollout gradxrollout lxt"}


echo "========================================="
echo "   Starting Sensitivity Evaluation      "
echo "========================================="


# 1. Define your configurations
# Add or remove paths here to change what the script runs
MODELS="configs/models/${MODEL_NAME}.yaml"

if [ ! -f "$MODELS" ]; then
    echo "ERROR : The file $MODEL_NAME does not exist !"
    exit 1
fi

DATASETS="configs/datasets/${DATASET_NAME}.yaml"

if [ ! -f "$DATASETS" ]; then
    echo "ERROR : The file $DATASET_NAME does not exist !"
    exit 1
fi

OUTPUT_DIR=(
    "logs/sensitivity"
)



# 2. Nested loops to run every combination
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        
        echo ""
        echo ">>> Launching Job: Model=$MODEL | Dataset=$DATASET"
        echo ">>> Explainers: $EXPLAINERS"
        echo ">>> Executing module: src.benchmarks.$DATASET_NAME"
        
        # 3. Execute the Python script
        python -m src.benchmarks.f_syn_sensitivity \
            --model_config "$MODEL" \
            --dataset_config "$DATASET" \
            --gpu_id $GPU_ID \
            --output_dir $OUTPUT_DIR \
            --explainers $EXPLAINERS \
            # --max_samples 2
            
        echo "<<< Job Finished. Moving to next configuration..."
        sleep 5 # Give the OS a few seconds to fully flush the GPU VRAM
        
    done
done

echo "========================================="
echo "   All Benchmarks Completed Successfully! "
echo "========================================="
