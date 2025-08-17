#!/bin/bash
#SBATCH -J TestJob
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>

# PTQ Experiment: qdrop + lsq + resnet50

# Activate MQBench environment
source /home/alz07xz/project/kmeans_results/MQBench/mqbench/bin/activate

echo "🚀 Starting PTQ Experiment: qdrop + lsq + resnet50"
echo "=========================================="

# Create results directory
mkdir -p results/qdrop_lsq_resnet50
cd results/qdrop_lsq_resnet50

# Fixed parameters
model="resnet50"
adv_mode="qdrop"
quant_model="lsq"
w_bits=8
a_bits=8
alpha=0.5
num_clusters=16
pca_dim=50
batch_size=48
calib_batches=24
logits_batches=8

echo "Parameters:"
echo "  Model: $model"
echo "  Advanced Mode: $adv_mode"
echo "  Quant Model: $quant_model"
echo "  Weight Bits: $w_bits"
echo "  Activation Bits: $a_bits"
echo "  Alpha: $alpha"
echo "  Clusters: $num_clusters"
echo "  PCA dim: $pca_dim"
echo "  Batch Size: $batch_size"
echo "  Calib Batches: $calib_batches"
echo "  Logits Batches: $logits_batches"
echo "  Alpha List: $alpha_list"
echo "  Clusters List: $num_clusters_list"
echo "  PCA Dim List: $pca_dim_list"
echo "=========================================="

# Create experiment output directory
exp_dir="qdrop_lsq_resnet50_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$exp_dir"

echo "🔄 Running experiment..."
echo "Time: $(date)"
echo "------------------------------------------"

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
    --alpha_list $alpha_list \
    --num_clusters_list $num_clusters_list \
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
    echo "✅ Experiment completed successfully!"
    echo "COMPLETED: $(date)" > "$exp_dir/experiment_completed.txt"
    echo "Results saved in: $exp_dir"
else
    echo "❌ Experiment failed!"
    echo "FAILED: $(date)" > "$exp_dir/experiment_failed.txt"
    echo "Check logs in: $exp_dir"
fi

echo "------------------------------------------"
echo "🎉 Experiment finished!"
echo "Results directory: $exp_dir"
