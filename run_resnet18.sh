#!/bin/bash

# Run MQBench PTQ experiments for ResNet18
# Loops through different alpha, num_clusters, and pca_dim values

echo "🚀 Starting ResNet18 PTQ experiments..."
echo "=========================================="

# Create results directory
mkdir -p results/resnet18
cd results/resnet18

# Define parameter lists
alphas=(0.2 0.4 0.6 0.8 1.0)
num_clusters=(8 16 32 64)
pca_dims=(25 50 100)

# Counter for experiments
exp_count=1
total_experiments=$((${#alphas[@]} * ${#num_clusters[@]} * ${#pca_dims[@]}))

echo "Total experiments: $total_experiments"
echo "Parameters:"
echo "  Alphas: ${alphas[*]}"
echo "  Clusters: ${num_clusters[*]}"
echo "  PCA dims: ${pca_dims[*]}"
echo "=========================================="

# Loop through all combinations
for alpha in "${alphas[@]}"; do
    for num_cluster in "${num_clusters[@]}"; do
        for pca_dim in "${pca_dims[@]}"; do
            echo ""
            echo "🔄 Experiment $exp_count/$total_experiments"
            echo "Parameters: alpha=$alpha, num_clusters=$num_cluster, pca_dim=$pca_dim"
            echo "Time: $(date)"
            echo "------------------------------------------"
            
            # Create experiment-specific output directory
            exp_dir="exp_${exp_count}_alpha${alpha}_clusters${num_cluster}_pca${pca_dim}"
            mkdir -p "$exp_dir"
            
            # Run the PTQ experiment
            python ../../mq_bench_ptq.py \
                --model resnet18 \
                --w_bits 8 \
                --a_bits 8 \
                --quant_model fixed \
                --alpha "$alpha" \
                --num_clusters "$num_cluster" \
                --pca_dim "$pca_dim" \
                --alpha_list $alpha_list \
                --num_clusters_list "$num_cluster" \
                --pca_dim_list $pca_dim_list \
                --batch_size 64 \
                --calib_batches 32 \
                --logits_batches 10 \
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
echo "🎉 All ResNet18 experiments completed!"
echo "Results saved in: $(pwd)"
echo "Total experiments run: $((exp_count - 1))"
