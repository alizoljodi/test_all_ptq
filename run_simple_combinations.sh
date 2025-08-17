#!/bin/bash

# Simple script to run all combinations of advanced modes, models, and quantization models
# Loops through: adv_mode × model × quant_model

echo "🚀 Starting Simple Combinations PTQ Experiments"
echo "=========================================="

# Create results directory
mkdir -p results/simple_combinations
cd results/simple_combinations

# Define parameter lists
adv_modes=("adaround" "brecq" "qdrop")
models=("resnet18" "resnet50" "mnasnet" "mobilenet_v2")
quant_models=("fixed" "learnable" "lsq" "lsqplus")

# Fixed parameters for simplicity
w_bits=8
a_bits=8
alpha=0.5
num_clusters=16
pca_dim=50

# Counter for experiments
exp_count=1
total_experiments=$((${#adv_modes[@]} * ${#models[@]} * ${#quant_models[@]}))

echo "Total experiments: $total_experiments"
echo "Parameters:"
echo "  Advanced Modes: ${adv_modes[*]}"
echo "  Models: ${models[*]}"
echo "  Quant Models: ${quant_models[*]}"
echo "  Fixed: w_bits=$w_bits, a_bits=$a_bits, alpha=$alpha, clusters=$num_clusters, pca_dim=$pca_dim"
echo "=========================================="

# Loop through all combinations
for adv_mode in "${adv_modes[@]}"; do
    for model in "${models[@]}"; do
        for quant_model in "${quant_models[@]}"; do
            echo ""
            echo "🔄 Experiment $exp_count/$total_experiments"
            echo "Parameters:"
            echo "  Advanced Mode: $adv_mode"
            echo "  Model: $model"
            echo "  Quant Model: $quant_model"
            echo "Time: $(date)"
            echo "------------------------------------------"
            
            # Set batch size based on model
            if [ "$model" = "resnet50" ]; then
                batch_size=48
                calib_batches=24
                logits_batches=8
            else
                batch_size=64
                calib_batches=32
                logits_batches=10
            fi
            
            # Create experiment-specific output directory
            exp_dir="exp_${exp_count}_${adv_mode}_${model}_${quant_model}"
            mkdir -p "$exp_dir"
            
            # Run the PTQ experiment
            python ../../mq_bench_ptq.py \
                --model "$model" \
                --w_bits "$w_bits" \
                --a_bits "$a_bits" \
                --quant_model "$quant_model" \
                --adv_mode "$adv_mode" \
                --alpha "$alpha" \
                --num_clusters "$num_clusters" \
                --pca_dim "$pca_dim" \
                --alpha_list "$alpha" \
                --num_clusters_list "$num_clusters" \
                --pca_dim_list "$pca_dim" \
                --batch_size "$batch_size" \
                --calib_batches "$calib_batches" \
                --logits_batches "$logits_batches" \
                --output_dir "$exp_dir" \
                --extract_logits \
                --check_train_structure \
                --save_csv \
                --log_file "$exp_dir/experiment.log" \
                --verbose
            
            # Check if experiment completed successfully
            if [ $? -eq 0 ]; then
                echo "✅ Experiment $exp_count completed successfully"
                echo "COMPLETED: $(date)" > "$exp_dir/experiment_completed.txt"
            else
                echo "❌ Experiment $exp_count failed"
                echo "FAILED: $(date)" > "$exp_dir/experiment_failed.txt"
            fi
            
            echo "------------------------------------------"
            exp_count=$((exp_count + 1))
        done
    done
done

echo ""
echo "🎉 All Simple Combinations experiments completed!"
echo "Results saved in: $(pwd)"
echo "Total experiments run: $((exp_count - 1))"
echo ""
echo "📊 Final Summary:"
echo "  Advanced Modes tested: ${#adv_modes[@]} (${adv_modes[*]})"
echo "  Models tested: ${#models[@]} (${models[*]})"
echo "  Quant Models tested: ${#quant_models[@]} (${quant_models[*]})"
echo "  Total combinations: $((exp_count - 1))"
echo ""
echo "📁 Results organized by:"
echo "  - Advanced PTQ technique"
echo "  - Model architecture"
echo "  - Quantization method"
