#!/bin/bash

# Set up logging
LOG_FILE="night_evaluation_$(date +%Y%m%d_%H%M%S).log"
echo "Starting night evaluation pipeline at $(date)" | tee -a "$LOG_FILE"

# Function to run command with logging
run_command() {
    local cmd="$1"
    local step_name="$2"
    
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Starting: $step_name at $(date)" | tee -a "$LOG_FILE"
    echo "Command: $cmd" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed: $step_name at $(date)" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed: $step_name at $(date) with exit code $exit_code" | tee -a "$LOG_FILE"
        exit $exit_code
    fi
    echo "" | tee -a "$LOG_FILE"
}

# Ensure we're using the right python
export PATH="/home/ubuntu/miniconda3/envs/omnidocbench/bin:$PATH"

# Step 1: Marker
run_command "python tools/model_infer/marker_img2md.py --omnidocbench_json ./OmniDocBench.json --images_dir ./images --output_dir ./night_results/marker_mds" "Marker Image to Markdown"

# Step 2: Unstructured Images
run_command "python tools/model_infer/unstructured_img2md.py --omnidocbench_json ./OmniDocBench.json --images_dir ./images --output_dir ./night_results/unstructured_imgs_mds --strategy auto" "Unstructured Images to Markdown"

# Step 3: Unstructured PDFs
run_command "python tools/pdf_to_markdown/pdf_to_md.py --omnidocbench_json ./OmniDocBench.json --pdf_dir ./ori_pdfs --output_dir ./night_results/unstructured_pdfs_mds" "PDF to Markdown"

echo "All processing steps completed. Starting validation..." | tee -a "$LOG_FILE"

# Step 4: Validations
run_command "python pdf_validation.py --config ./configs/nightmarker.yaml" "Marker Validation"

run_command "python pdf_validation.py --config ./configs/nightunstructimgs.yaml" "Unstructured Images Validation"

run_command "python pdf_validation.py --config ./configs/nightunstructpdfs.yaml" "Unstructured PDFs Validation"

echo "========================================" | tee -a "$LOG_FILE"
echo "All steps completed successfully at $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE" 