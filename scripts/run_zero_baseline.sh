#!/bin/bash

# Stop the script immediately if any command fails
set -e 

GPU_ID=${1:-0}
MODEL_NAME=${2:-"internvl_2b"}
DATASET_NAME=${3:-"mmstar"}
EXPLAINERS=${4:-"random gradcam tam llavacam inputxgradients integratedgradients rollout gradxrollout lxt"}



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

DATASETS="configs/datasets/${DATASET_NAME}.yaml"

if [ ! -f "$DATASETS" ]; then
    echo "ERROR : The file $MODEL_CONFIG does not exist !"
    exit 1
fi

OUTPUT_DIR=(
    "logs/zero_baseline"
)

# Pass all explainers as a single string separated by spaces
# #EXPLAINERS="gradcam lxt integratedgradients"
# EXPLAINERS="tam inputxgradients gradxrollout " 
# #EXPLAINERS="random llavacam rollout"


# 2. Nested loops to run every combination
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        
        echo ""
        echo ">>> Launching Job: Model=$MODEL | Dataset=$DATASET"
        echo ">>> Explainers: $EXPLAINERS"
        echo ">>> Executing module: src.benchmarks.$DATASET_NAME"
        
        # 3. Execute the Python script
        # Notice we omitted --max_samples so it runs the whole dataset!
        #python -m src.benchmarks.repope \
        python -m src.benchmarks."${DATASET_NAME}" \
            --model_config "$MODEL" \
            --dataset_config "$DATASET" \
            --gpu_id $GPU_ID \
            --output_dir $OUTPUT_DIR \
            --explainers $EXPLAINERS \
            -- zero-state
            #--max_samples 2
            
        echo "<<< Job Finished. Moving to next configuration..."
        sleep 5 # Give the OS a few seconds to fully flush the GPU VRAM
        
    done
done

echo "========================================="
echo "   All Benchmarks Completed Successfully! "
echo "========================================="
