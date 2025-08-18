#!/bin/bash
#SBATCH --job-name=mqbench_ptq_experiments
#SBATCH --output=logs/mqbench_ptq_%A_%a.out
#SBATCH --error=logs/mqbench_ptq_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1919%1  # 1920 total jobs, max 8 concurrent
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com

source /home/alz07xz/project/kmeans_results/MQBench/mqbench/bin/activate

# Create logs directory
mkdir -p logs
mkdir -p results

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Load modules (adjust for your system)
module load cuda/11.8
module load python/3.9

# Activate conda environment (adjust path)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mqbench_env

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
total_models=${#models[@]}
total_adv_modes=${#adv_modes[@]}
total_w_bits=${#w_bits[@]}
total_a_bits=${#a_bits[@]}
total_quant_models=${#quant_models[@]}
total_alphas=${#alphas[@]}
total_num_clusters=${#num_clusters[@]}
total_pca_dims=${#pca_dims[@]}

# Calculate indices for current job
job_id=$SLURM_ARRAY_TASK_ID

# Calculate parameter indices
pca_dim_idx=$((job_id % total_pca_dims))
job_id=$((job_id / total_pca_dims))

num_clusters_idx=$((job_id % total_num_clusters))
job_id=$((job_id / total_num_clusters))

alpha_idx=$((job_id % total_alphas))
job_id=$((job_id / total_alphas))

quant_model_idx=$((job_id % total_quant_models))
job_id=$((job_id / total_quant_models))

a_bits_idx=$((job_id % total_a_bits))
job_id=$((job_id / total_a_bits))

w_bits_idx=$((job_id % total_w_bits))
job_id=$((job_id / total_w_bits))

adv_mode_idx=$((job_id % total_adv_modes))
job_id=$((job_id / total_adv_modes))

model_idx=$((job_id % total_models))

# Get actual parameter values
model=${models[$model_idx]}
adv_mode=${adv_modes[$adv_mode_idx]}
w_bit=${w_bits[$w_bits_idx]}
a_bit=${a_bits[$a_bits_idx]}
quant_model=${quant_models[$quant_model_idx]}
alpha=${alphas[$alpha_idx]}
num_cluster=${num_clusters[$num_clusters_idx]}
pca_dim=${pca_dims[$pca_dim_idx]}

# Create unique output directory for this experiment
timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="${model}_${adv_mode}_w${w_bit}a${a_bit}_${quant_model}_a${alpha}_c${num_cluster}_pca${pca_dim}"
output_dir="results/${exp_name}_${timestamp}"

# Create experiment-specific log file
log_file="logs/${exp_name}_${timestamp}.log"

echo "=========================================="
echo "Starting PTQ Experiment: $exp_name"
echo "Job ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo "Parameters:"
echo "  Model: $model"
echo "  Advanced Mode: $adv_mode"
echo "  Weight Bits: $w_bit"
echo "  Activation Bits: $a_bit"
echo "  Quantization Model: $quant_model"
echo "  Alpha: $alpha"
echo "  Number of Clusters: $num_cluster"
echo "  PCA Dimension: $pca_dim"
echo "  Output Directory: $output_dir"
echo "=========================================="

# Set smaller batch sizes for lower bit models to avoid memory issues
if [ $w_bit -eq 2 ] || [ $a_bit -eq 2 ]; then
    batch_size=32
    calib_batches=16
elif [ $w_bit -eq 4 ] || [ $a_bit -eq 4 ]; then
    batch_size=48
    calib_batches=24
else
    batch_size=64
    calib_batches=32
fi

# Set smaller logits batches for memory efficiency
logits_batches=5

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
    --output_dir $output_dir \
    --extract_logits \
    --check_train_structure \
    --save_csv \
    --log_file $log_file \
    --verbose

# Check if the experiment completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Experiment $exp_name completed successfully!"
    echo "Results saved to: $output_dir"
    echo "=========================================="
    
    # Create a completion marker
    echo "COMPLETED: $(date)" > "$output_dir/experiment_completed.txt"
    
    # Optional: Copy results to a central location
    cp -r "$output_dir" "results/completed/"
    
else
    echo "=========================================="
    echo "Experiment $exp_name FAILED!"
    echo "Check logs: $log_file"
    echo "=========================================="
    
    # Create a failure marker
    echo "FAILED: $(date)" > "$output_dir/experiment_failed.txt"
    
    # Exit with error code
    exit 1
fi

echo "Job $SLURM_ARRAY_TASK_ID finished at $(date)"
