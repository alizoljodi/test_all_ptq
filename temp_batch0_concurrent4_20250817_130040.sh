#!/bin/bash
#SBATCH --job-name=mqbench_ptq_experiments
#SBATCH --output=logs/mqbench_ptq_%j.out
#SBATCH --error=logs/mqbench_ptq_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com

# Batch-specific parameters
export BATCH_ID=0
export START_IDX=0
export END_IDX=6480
export TOTAL_EXPERIMENTS=6480


# Batch-specific parameters (set by submit_experiments.py)
export BATCH_ID=${BATCH_ID:-0}
export START_IDX=${START_IDX:-0}
export END_IDX=${END_IDX:-480}
export TOTAL_EXPERIMENTS=${TOTAL_EXPERIMENTS:-480}

echo "=========================================="
echo "MQBench PTQ Experiments - Batch $BATCH_ID"
echo "=========================================="
echo "Batch ID: $BATCH_ID"
echo "Start Index: $START_IDX"
echo "End Index: $END_IDX"
echo "Total Experiments in this batch: $TOTAL_EXPERIMENTS"
echo "=========================================="

# Load modules and activate environment
source /home/alz07xz/project/kmeans_results/MQBench/mqbench/bin/activate

# Create output directory for this batch
output_dir="results/batch_${BATCH_ID}_${START_IDX}_${END_IDX}"
mkdir -p $output_dir
mkdir -p logs

# Define parameter arrays
models=("resnet18" "resnet50" "mnasnet" "mobilenet_v2")
adv_modes=("adaround" "brecq" "qdrop")
w_bits=(2 4 8)
a_bits=(2 4 8)
quant_models=("fixed" "learnable" "lsq" "lsqplus")
alphas=(0.2 0.4 0.6 0.8 1.0)
num_clusters=(8 16 32 64)
pca_dims=(25 50 100)

# Calculate total combinations
total_combinations=1920

# Calculate which experiments this batch should run
echo "Processing experiments $START_IDX to $((END_IDX-1)) of $total_combinations total..."

# Run experiments for this batch
experiment_count=0
for model in "${models[@]}"; do
    for adv_mode in "${adv_modes[@]}"; do
        for w_bit in "${w_bits[@]}"; do
            for a_bit in "${a_bits[@]}"; do
                for quant_model in "${quant_models[@]}"; do
                    for alpha in "${alphas[@]}"; do
                        for num_cluster in "${num_clusters[@]}"; do
                            for pca_dim in "${pca_dims[@]}"; do
                                # Check if this experiment belongs to this batch
                                if [ $experiment_count -ge $START_IDX ] && [ $experiment_count -lt $END_IDX ]; then
                                    echo "Running experiment $experiment_count: $model $adv_mode w${w_bit}a${a_bit} $quant_model a${alpha} c${num_cluster} pca${pca_dim}"
                                    
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
                                    exp_dir="${output_dir}/exp_${experiment_count}_${model}_${adv_mode}_w${w_bit}a${a_bit}_${quant_model}_a${alpha}_c${num_cluster}_pca${pca_dim}"
                                    mkdir -p "$exp_dir"
                                    
                                    # Run the PTQ experiment
                                    python mq_bench_ptq.py \
                                        --model $model \
                                        --adv_mode $adv_mode \
                                        --w_bits $w_bit \
                                        --a_bits $a_bit \
                                        --quant_model $quant_model \
                                        --alpha $alpha \
                                        --num_clusters $num_cluster \
                                        --pca_dim $pca_dim \
                                        --alpha_list $alpha \
                                        --num_clusters_list $num_cluster \
                                        --pca_dim_list $pca_dim \
                                        --batch_size $batch_size \
                                        --calib_batches $calib_batches \
                                        --logits_batches $logits_batches \
                                        --output_dir "$exp_dir" \
                                        --extract_logits \
                                        --check_train_structure \
                                        --save_csv \
                                        --log_file "$exp_dir/experiment.log" \
                                        --verbose
                                    
                                    # Check if experiment completed successfully
                                    if [ $? -eq 0 ]; then
                                        echo "✅ Experiment $experiment_count completed successfully" > "$exp_dir/experiment_completed.txt"
                                        echo "Experiment $experiment_count completed at $(date)" >> "$exp_dir/experiment_completed.txt"
                                    else
                                        echo "❌ Experiment $experiment_count failed" > "$exp_dir/experiment_failed.txt"
                                        echo "Experiment $experiment_count failed at $(date)" >> "$exp_dir/experiment_failed.txt"
                                    fi
                                fi
                                
                                experiment_count=$((experiment_count + 1))
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "=========================================="
echo "Batch $BATCH_ID completed!"
echo "Processed experiments $START_IDX to $((END_IDX-1))"
echo "Results saved to: $output_dir"
echo "=========================================="

echo "Job $SLURM_JOB_ID finished at $(date)"
