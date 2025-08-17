#!/bin/bash

# Run MQBench PTQ experiments for ResNet18 with advanced quantization options
# Loops through different quantization models, advanced methods, bit widths, and PCA parameters

echo "🚀 Starting ResNet18 Advanced PTQ experiments..."
echo "=========================================="

# Create results directory
mkdir -p results/resnet18_advanced
cd results/resnet18_advanced

# Define parameter lists
quant_models=("fixed" "learnable" "lsq" "lsqplus")
adv_modes=("adaround" "brecq" "qdrop")
w_bits=(2 4 8)
a_bits=(2 4 8)
alphas=(0.2 0.4 0.6 0.8 1.0)
num_clusters=(8 16 32 64)
pca_dims=(25 50 100)

# Counter for experiments
exp_count=1
total_experiments=$((${#quant_models[@]} * ${#adv_modes[@]} * ${#w_bits[@]} * ${#a_bits[@]} * ${#alphas[@]} * ${#num_clusters[@]} * ${#pca_dims[@]}))

echo "Total experiments: $total_experiments"
echo "Parameters:"
echo "  Quant Models: ${quant_models[*]}"
echo "  Advanced Modes: ${adv_modes[*]}"
echo "  Weight Bits: ${w_bits[*]}"
echo "  Activation Bits: ${a_bits[*]}"
echo "  Alphas: ${alphas[*]}"
echo "  Clusters: ${num_clusters[*]}"
echo "  PCA dims: ${pca_dims[*]}"
echo "=========================================="

# Loop through all combinations
for quant_model in "${quant_models[@]}"; do
    for adv_mode in "${adv_modes[@]}"; do
        for w_bit in "${w_bits[@]}"; do
            for a_bit in "${a_bits[@]}"; do
                for alpha in "${alphas[@]}"; do
                    for num_cluster in "${num_clusters[@]}"; do
                        for pca_dim in "${pca_dims[@]}"; do
                            echo ""
                            echo "🔄 Experiment $exp_count/$total_experiments"
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
                            
                            # Set batch size and calibration batches based on bit width
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
                            
                            # Create experiment-specific output directory
                            exp_dir="exp_${exp_count}_${quant_model}_${adv_mode}_w${w_bit}a${a_bit}_alpha${alpha}_clusters${num_cluster}_pca${pca_dim}"
                            mkdir -p "$exp_dir"
                            
                            # Run the PTQ experiment
                            python ../../mq_bench_ptq.py \
                                --model resnet18 \
                                --w_bits "$w_bit" \
                                --a_bits "$a_bit" \
                                --quant_model "$quant_model" \
                                --adv_mode "$adv_mode" \
                                --alpha "$alpha" \
                                --num_clusters "$num_cluster" \
                                --pca_dim "$pca_dim" \
                                --alpha_list "$alpha" \
                                --num_clusters_list "$num_cluster" \
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
    done
done

echo ""
echo "🎉 All ResNet18 Advanced experiments completed!"
echo "Results saved in: $(pwd)"
echo "Total experiments run: $((exp_count - 1))"
echo ""
echo "📊 Experiment Summary:"
echo "  Quant Models tested: ${#quant_models[@]}"
echo "  Advanced Modes tested: ${#adv_modes[@]}"
echo "  Weight bit widths tested: ${#w_bits[@]}"
echo "  Activation bit widths tested: ${#a_bits[@]}"
echo "  Alpha values tested: ${#alphas[@]}"
echo "  Cluster numbers tested: ${#num_clusters[@]}"
echo "  PCA dimensions tested: ${#pca_dims[@]}"
echo "  Total combinations: $((exp_count - 1))"
