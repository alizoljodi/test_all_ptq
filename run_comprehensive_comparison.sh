#!/bin/bash

# Comprehensive PTQ comparison script
# Tests basic vs advanced quantization with different models and parameters

echo "🚀 Starting Comprehensive PTQ Comparison Experiments"
echo "=========================================="

# Create results directory
mkdir -p results/comprehensive_comparison
cd results/comprehensive_comparison

# Define parameter lists
models=("resnet18" "resnet50" "mnasnet0_5" "mobilenet_v2")
quant_models=("fixed" "learnable" "lsq" "lsqplus")
adv_modes=("adaround" "brecq" "qdrop")
w_bits=(2 4 8)
a_bits=(2 4 8)
alphas=(0.2 0.4 0.6 0.8 1.0)
num_clusters=(8 16 32 64)
pca_dims=(25 50 100)

# Counter for experiments
exp_count=1

echo "📊 Comprehensive Experiment Plan:"
echo "  Models: ${models[*]}"
echo "  Quant Models: ${quant_models[*]}"
echo "  Advanced Modes: ${adv_modes[*]}"
echo "  Weight Bits: ${w_bits[*]}"
echo "  Activation Bits: ${a_bits[*]}"
echo "  Alphas: ${alphas[*]}"
echo "  Clusters: ${num_clusters[*]}"
echo "  PCA dims: ${pca_dims[*]}"
echo "=========================================="

# Function to calculate total experiments
calculate_total() {
    local total=0
    for model in "${models[@]}"; do
        for quant_model in "${quant_models[@]}"; do
            for adv_mode in "${adv_modes[@]}"; do
                for w_bit in "${w_bits[@]}"; do
                    for a_bit in "${a_bits[@]}"; do
                        for alpha in "${alphas[@]}"; do
                            for num_cluster in "${num_clusters[@]}"; do
                                for pca_dim in "${pca_dims[@]}"; do
                                    total=$((total + 1))
                                done
                            done
                        done
                    done
                done
            done
        done
    done
    echo $total
}

total_experiments=$(calculate_total)
echo "Total experiments: $total_experiments"
echo "=========================================="

# Loop through all combinations
for model in "${models[@]}"; do
    for quant_model in "${quant_models[@]}"; do
        for adv_mode in "${adv_modes[@]}"; do
            for w_bit in "${w_bits[@]}"; do
                for a_bit in "${a_bits[@]}"; do
                    for alpha in "${alphas[@]}"; do
                        for num_cluster in "${num_clusters[@]}"; do
                            for pca_dim in "${pca_dims[@]}"; do
                                echo ""
                                echo "🔄 Experiment $exp_count/$total_experiments"
                                echo "Model: $model"
                                echo "Parameters:"
                                echo "  Quant Model: $quant_model"
                                echo "  Advanced Mode: $adv_mode"
                                echo "  Weight Bits: $w_bit"
                                echo "  Activation Bits: $a_bit"
                                echo "  Alpha: $alpha"
                                echo "  Clusters: $num_cluster"
                                echo "  PCA dim: $pca_dim"
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

# Parameter lists for looping (space-separated values)
alpha_list="0.2 0.4 0.6 0.8 1.0"
num_clusters_list="8 16 32 64"
pca_dim_list="25 50 100"
                                fi
                                
                                # Reduce batch size for larger models
                                if [ "$model" = "resnet50" ]; then
                                    batch_size=$((batch_size * 3 / 4))
                                    calib_batches=$((calib_batches * 3 / 4))
                                    logits_batches=$((logits_batches * 4 / 5))
                                fi
                                
                                # Create experiment-specific output directory
                                exp_dir="exp_${exp_count}_${model}_${quant_model}_${adv_mode}_w${w_bit}a${a_bit}_alpha${alpha}_clusters${num_cluster}_pca${pca_dim}"
                                mkdir -p "$exp_dir"
                                
                                # Run the PTQ experiment
                                python ../../mq_bench_ptq.py \
                                    --model "$model" \
                                    --w_bits "$w_bit" \
                                    --a_bits "$a_bit" \
                                    --quant_model "$quant_model" \
                                    --adv_mode "$adv_mode" \
                                    --alpha "$alpha" \
                                    --num_clusters "$num_cluster" \
                                    --pca_dim "$pca_dim" \
                                    --alpha_list $alpha_list \
                                    --num_clusters_list "$num_cluster" \
                                    --pca_dim_list $pca_dim_list \
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
        done
    done
done

echo ""
echo "🎉 All Comprehensive experiments completed!"
echo "Results saved in: $(pwd)"
echo "Total experiments run: $((exp_count - 1))"
echo ""
echo "📊 Final Experiment Summary:"
echo "  Models tested: ${#models[@]} (${models[*]})"
echo "  Quant Models tested: ${#quant_models[@]} (${quant_models[*]})"
echo "  Advanced Modes tested: ${#adv_modes[@]} (${adv_modes[*]})"
echo "  Weight bit widths tested: ${#w_bits[@]} (${w_bits[*]})"
echo "  Activation bit widths tested: ${#a_bits[@]} (${a_bits[*]})"
echo "  Alpha values tested: ${#alphas[@]} (${alphas[*]})"
echo "  Cluster numbers tested: ${#num_clusters[@]} (${num_clusters[*]})"
echo "  PCA dimensions tested: ${#pca_dims[@]} (${pca_dims[*]})"
echo "  Total combinations: $((exp_count - 1))"
echo ""
echo "📁 Results organized by:"
echo "  - Model type"
echo "  - Quantization method"
echo "  - Advanced PTQ technique"
echo "  - Bit precision"
echo "  - PCA clustering parameters"
