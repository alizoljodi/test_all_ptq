#!/usr/bin/env python3

import os

# Define parameter lists
adv_modes = ["adaround", "brecq", "qdrop"]
models = ["resnet18", "resnet50", "mnasnet0_5", "mobilenet_v2"]
quant_models = ["fixed", "learnable", "lsq", "lsqplus"]

# Template for bash files
bash_template = '''#!/bin/bash
#SBATCH -J TestJob
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>

# PTQ Experiment: {adv_mode} + {quant_model} + {model}

# Activate MQBench environment
source /home/alz07xz/project/kmeans_results/MQBench/mqbench/bin/activate

echo "🚀 Starting PTQ Experiment: {adv_mode} + {quant_model} + {model}"
echo "=========================================="

# Create results directory
mkdir -p results/{adv_mode}_{quant_model}_{model}
cd results/{adv_mode}_{quant_model}_{model}

# Fixed parameters
model="{model}"
adv_mode="{adv_mode}"
quant_model="{quant_model}"
w_bits=8
a_bits=8
alpha=0.5
num_clusters=16
pca_dim=50
batch_size={batch_size}
calib_batches={calib_batches}
logits_batches={logits_batches}

# Parameter lists for looping (space-separated values)
alpha_list="0.2 0.4 0.6 0.8 1.0"
num_clusters_list="8 16 32 64"
pca_dim_list="25 50 100"

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
exp_dir="{adv_mode}_{quant_model}_{model}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$exp_dir"

echo "🔄 Running experiment..."
echo "Time: $(date)"
echo "------------------------------------------"

# Run the PTQ experiment
python ../../mq_bench_ptq.py \\
    --model "$model" \\
    --w_bits "$w_bits" \\
    --a_bits "$a_bits" \\
    --quant_model "$quant_model" \\
    --adv_mode "$adv_mode" \\
    --alpha "$alpha" \\
    --num_clusters "$num_clusters" \\
    --pca_dim "$pca_dim" \\
    --alpha_list $alpha_list \\
    --num_clusters_list $num_clusters_list \\
    --pca_dim_list $pca_dim_list \\
    --batch_size "$batch_size" \\
    --calib_batches "$calib_batches" \\
    --logits_batches "$logits_batches" \\
    --output_dir "$exp_dir" \\
    --extract_logits \\
    --check_train_structure \\
    --save_csv \\
    --log_file "$exp_dir/experiment.log" \\
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
'''

def get_batch_settings(model):
    """Get batch size settings based on model"""
    if model == "resnet50":
        return 48, 24, 8
    elif model == "mnasnet0_5":
        return 64, 32, 10
    else:
        return 64, 32, 10

def main():
    print("🔧 Creating all combination bash files...")
    print("=" * 50)
    
    file_count = 0
    
    for adv_mode in adv_modes:
        for model in models:
            for quant_model in quant_models:
                # Handle special case for MNASNet to maintain readable filenames
                if model == "mnasnet0_5":
                    filename = f"{adv_mode}_{quant_model}_mnasnet.sh"
                else:
                    filename = f"{adv_mode}_{quant_model}_{model}.sh"
                
                # Get batch settings
                batch_size, calib_batches, logits_batches = get_batch_settings(model)
                
                # Create file content
                content = bash_template.format(
                    adv_mode=adv_mode,
                    model=model,
                    quant_model=quant_model,
                    batch_size=batch_size,
                    calib_batches=calib_batches,
                    logits_batches=logits_batches
                )
                
                # Write file
                with open(filename, 'w') as f:
                    f.write(content)
                
                # Make executable
                os.chmod(filename, 0o755)
                
                print(f"✅ Created: {filename}")
                file_count += 1
    
    print("")
    print("🎉 Generated all {file_count} combination files!")
    print("=" * 50)
    print("Files created:")
    print("")
    
    # List all created files
    for adv_mode in adv_modes:
        for model in models:
            for quant_model in quant_models:
                # Handle special case for MNASNet to maintain readable filenames
                if model == "mnasnet0_5":
                    filename = f"{adv_mode}_{quant_model}_mnasnet.sh"
                else:
                    filename = f"{adv_mode}_{quant_model}_{model}.sh"
                print(f"  - {filename}")
    
    print("")
    print("🚀 To run any experiment:")
    print("  ./adaround_fixed_resnet18.sh")
    print("  ./brecq_learnable_resnet50.sh")
    print("  ./qdrop_lsq_mnasnet.sh")
    print("  # etc...")
    print("")
    print("📁 Results will be saved in: results/[combination_name]/")

if __name__ == "__main__":
    main()
