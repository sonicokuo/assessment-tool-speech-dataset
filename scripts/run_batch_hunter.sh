#!/bin/bash

# Configuration
GLOBAL_START_ROW=4170
NUM_BATCHES=139
ROWS_PER_BATCH=10

SPLIT="train-100"
INPUT_CSV="$SHARED/data/features/${SPLIT}.csv"
OUTPUT_CSV="$SHARED/data/verbalized/${SPLIT}.csv"

echo "Starting batched processing for ${SPLIT}..."

for ((i=0; i<NUM_BATCHES; i++)); do
    START_ROW=$((GLOBAL_START_ROW + i * ROWS_PER_BATCH))
    
    echo "=========================================================="
    echo "Processing batch $((i+1))/${NUM_BATCHES} (start_row=${START_ROW}, num_rows=${ROWS_PER_BATCH})"
    echo "=========================================================="
    
    python scripts/feature_verbalization_custom.py \
        --input "$INPUT_CSV" \
        --output "$OUTPUT_CSV" \
        --start_row "$START_ROW" \
        --num_rows "$ROWS_PER_BATCH"
        
    # Check if the python script failed (e.g., if Ollama completely crashed)
    if [ $? -ne 0 ]; then
        echo "Error detected at start_row=${START_ROW}. Exiting loop."
        echo "You can resume later by starting the loop from i=${i}."
        exit 1
    fi
done

echo "All batches completed!"
