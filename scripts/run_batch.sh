#!/bin/bash

# Configuration
# You can provide start_row and end_row as command-line arguments:
# e.g., ./scripts/run_batched.sh 0 1390
# Otherwise, it uses the defaults below:
START_ROW=${1:-0}
END_ROW=${2:-1390}
ROWS_PER_BATCH=10

SPLIT="train-100"
INPUT_CSV="$SHARED/data/features/${SPLIT}.csv"
OUTPUT_CSV="$SHARED/data/verbalized/${SPLIT}_batches/${SPLIT}.csv"

echo "Starting batched processing for ${SPLIT} from row ${START_ROW} to ${END_ROW}..."

for ((row=START_ROW; row<END_ROW; row+=ROWS_PER_BATCH)); do
    # Ensure the last batch doesn't exceed END_ROW
    REMAINING=$((END_ROW - row))
    if [ $REMAINING -lt $ROWS_PER_BATCH ]; then
        BATCH_SIZE=$REMAINING
    else
        BATCH_SIZE=$ROWS_PER_BATCH
    fi
    
    echo "=========================================================="
    echo "Processing rows [${row} to $((row + BATCH_SIZE))) out of ${END_ROW}"
    echo "=========================================================="
    
    python scripts/feature_verbalization_custom.py \
        --input "$INPUT_CSV" \
        --output "$OUTPUT_CSV" \
        --start_row "$row" \
        --num_rows "$BATCH_SIZE"
        
    # Check if the python script failed (e.g., if Ollama completely crashed)
    if [ $? -ne 0 ]; then
        echo "Error detected at start_row=${row}. Exiting loop."
        echo "You can resume by running: ./scripts/run_batched.sh $row $END_ROW"
        exit 1
    fi
done

echo "All batches from ${START_ROW} to ${END_ROW} completed!"
