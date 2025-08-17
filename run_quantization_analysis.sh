#!/bin/bash

# Focused Quantization Analysis Script
# Tests specific scenarios to understand quantization impact

echo "🔬 Starting Focused Quantization Analysis"
echo "=========================================="

# Create results directory
mkdir -p results/quantization_analysis
cd results/quantization_analysis

# Define focused parameter combinations
models=("resnet18" "resnet50")
quant_models=("fixed" "learnable" "lsq" "lsqplus")
adv_modes=("adaround" "brecq" "qdrop")
w_bits=(2 4 8)
a_bits=(2 4 8)

# Fixed PCA parameters for focused analysis
alpha=0.5
num_clusters=16
pca_dim=50

# Counter for experiments
exp_count=1
total_experiments=$((${#models[@]} * ${#quant_models[@]} * ${#adv_modes[@]} * ${#w_bits[@]} * ${#a_bits[@]}))

echo "📊 Focused Analysis Plan:"
echo "  Models: ${models[*]}"
echo "  Quant Models: ${quant_models[*]}"
echo "  Advanced Modes: ${adv_modes[*]}"
echo "  Weight Bits: ${w_bits[*]}"
echo "  Activation Bits: ${a_bits[*]}"
echo "  Fixed PCA: alpha=$alpha, clusters=$num_clusters, pca_dim=$pca_dim"
echo "=========================================="
echo "Total experiments: $total_experiments"
echo "=========================================="

# Loop through focused combinations
for model in "${models[@]}"; do
    for quant_model in "${quant_models[@]}"; do
        for adv_mode in "${adv_modes[@]}"; do
            for w_bit in "${w_bits[@]}"; do
                for a_bit in "${a_bits[@]}"; do
                    echo ""
                    echo "🔄 Experiment $exp_count/$total_experiments"
                    echo "Model: $model"
                    echo "Parameters:"
                    echo "  Quant Model: $quant_model"
                    echo "  Advanced Mode: $adv_mode"
                    echo "  Weight Bits: $w_bit"
                    echo "  Activation Bits: $a_bit"
                    echo "  Fixed PCA: alpha=$alpha, clusters=$num_clusters, pca_dim=$pca_dim"
                    echo "Time: $(date)"
                    echo "------------------------------------------"
                    
                    # Set batch size and calibration batches based on bit width and model
                    if [ $w_bit -eq 2 ]; then
                        batch_size=32
                        calib_batches=16
                        logits_batches=5
                    elif [ $w_bit -eq 4 ]; then
                        batch_size=48
                        calib_batches=24
                        logits_batches=8
                    else
                        batch_size=64
                        calib_batches=32
                        logits_batches=10
                    fi
                    
                    # Reduce batch size for larger models
                    if [ "$model" = "resnet50" ]; then
                        batch_size=$((batch_size * 3 / 4))
                        calib_batches=$((calib_batches * 3 / 4))
                        logits_batches=$((logits_batches * 4 / 5))
                    fi
                    
                    # Create experiment-specific output directory
                    exp_dir="exp_${exp_count}_${model}_${quant_model}_${adv_mode}_w${w_bit}a${a_bit}_fixedPCA"
                    mkdir -p "$exp_dir"
                    
                    # Run the PTQ experiment
                    python ../../mq_bench_ptq.py \
                        --model "$model" \
                        --w_bits "$w_bit" \
                        --a_bits "$a_bit" \
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
    done
done

echo ""
echo "🎉 All Quantization Analysis experiments completed!"
echo "Results saved in: $(pwd)"
echo "Total experiments run: $((exp_count - 1))"
echo ""
echo "📊 Analysis Summary:"
echo "  This focused analysis tests:"
echo "  - Different quantization models (fixed, learnable, LSQ, LSQ+)"
echo "  - Advanced PTQ techniques (AdaRound, BRECQ, QDrop)"
echo "  - Various bit precisions (2, 4, 8 bits)"
echo "  - Fixed PCA parameters for fair comparison"
echo ""
echo "🔍 Key Questions Answered:"
echo "  1. How do different quantization models affect accuracy?"
echo "  2. Which advanced PTQ method works best for each model?"
echo "  3. What's the accuracy vs bit-width trade-off?"
echo "  4. How do different models respond to quantization?"
echo ""
echo "📁 Results organized by:"
echo "  - Model architecture"
echo "  - Quantization method"
echo "  - Advanced PTQ technique"
echo "  - Bit precision"
