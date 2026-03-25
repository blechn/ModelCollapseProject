#!/bin/bash

# Define the models and the experiment modes
MODELS=("pixelcnn")
EXPERIMENTS=("full" "replace" "add")

USE_FASHION=false
# Parse command line arguments for the BASH script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--fashion) USE_FASHION=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Prepare the fashion flag for the python call
FASHION_FLAG=""
if [ "$USE_FASHION" = true ]; then
    FASHION_FLAG="--fashion"
    DATASET_LABEL="fashion"
    echo "Running experiments with FASHION MNIST"
else
    DATASET_LABEL="mnist"
    echo "Running experiments with NORMAL MNIST"
fi

# Create a logs directory
mkdir -p run_pixelcnn_experiment_logs
echo "====================================================================="
echo "                  Starting PixelCNN experiment run                   "
echo "====================================================================="

for MODEL in "${MODELS[@]}"; do
    for EXP in "${EXPERIMENTS[@]}"; do
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        LOG_FILE="run_pixelcnn_experiment_logs/${MODEL}_${EXP}_run_${TIMESTAMP}.log"
        echo "=========================================================="
        echo "RUNNING: Model: $MODEL | Experiment: $EXP"
        echo "LOGGING TO: $LOG_FILE"
        echo "=========================================================="
        
        # Note: Ensure you updated full_experiment.py to accept "pixelcnn"
        python -m src.routines.full_experiment \
            --modelcls "$MODEL" \
            --experiment "$EXP" \
            --collapse_epochs 10 \
            --max_epochs 15 \
            --add_percentage 0.2 \
            --replace_percentage 0.2 \
            $FASHION_FLAG 2>&1 | tee "$LOG_FILE"
            
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "ERROR: $MODEL experiment $EXP exited with an error. Continuing!" | tee -a "$LOG_FILE"
        else
            echo "COMPLETED: $MODEL: $EXP" | tee -a "$LOG_FILE"
        fi
    done
done

echo "=========================================================="
echo "All experiments finished successfully."
echo "Shutting down system in 1 minute..."
echo "=========================================================="

shutdown -P +1 "Experiments Complete. System going down."
