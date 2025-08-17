#!/bin/bash

# Script to generate all individual combination bash files

echo "🔧 Generating all combination bash files..."
echo "=========================================="

# Define parameter lists
adv_modes=("adaround" "brecq" "qdrop")
models=("resnet18" "resnet50" "mnasnet0_5" "mobilenet_v2")
quant_models=("fixed" "learnable" "lsq" "lsqplus")

# Counter for files created
file_count=0

# Generate all combinations
for adv_mode in "${adv_modes[@]}"; do
    for model in "${models[@]}"; do
        for quant_model in "${quant_models[@]}"; do
            filename="${adv_mode}_${quant_model}_${model}.sh"
            
            # Set batch size based on model
            if [ "$model" = "resnet50" ]; then
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
            
            # Create the bash file
            cat > "$filename" << EOF
#!/bin/bash

# PTQ Experiment: $adv_mode + $quant_model + $model

echo "🚀 Starting PTQ Experiment: $adv_mode + $quant_model + $model"
echo "=========================================="

# Create results directory
mkdir -p results/${adv_mode}_${quant_model}_${model}
cd results/${adv_mode}_${quant_model}_${model}

# Fixed parameters
model="$model"
adv_mode="$adv_mode"
quant_model="$quant_model"
w_bits=8
a_bits=8
alpha=0.5
num_clusters=16
pca_dim=50
batch_size=$batch_size
calib_batches=$calib_batches
logits_batches=$logits_batches

echo "Parameters:"
echo "  Model: \$model"
echo "  Advanced Mode: \$adv_mode"
echo "  Quant Model: \$quant_model"
echo "  Weight Bits: \$w_bits"
echo "  Activation Bits: \$a_bits"
echo "  Alpha: \$alpha"
echo "  Clusters: \$num_clusters"
echo "  PCA dim: \$pca_dim"
echo "  Batch Size: \$batch_size"
echo "  Calib Batches: \$calib_batches"
echo "  Logits Batches: \$logits_batches"
echo "=========================================="

# Create experiment output directory
exp_dir="${adv_mode}_${quant_model}_${model}_\$(date +%Y%m%d_%H%M%S)"
mkdir -p "\$exp_dir"

echo "🔄 Running experiment..."
echo "Time: \$(date)"
echo "------------------------------------------"

# Run the PTQ experiment
python ../../mq_bench_ptq.py \\
    --model "\$model" \\
    --w_bits "\$w_bits" \\
    --a_bits "\$a_bits" \\
    --quant_model "\$quant_model" \\
    --adv_mode "\$adv_mode" \\
    --alpha "\$alpha" \\
    --num_clusters "\$num_clusters" \\
    --pca_dim "\$pca_dim" \\
    --alpha_list "\$alpha" \\
    --num_clusters_list "\$num_clusters" \\
    --pca_dim_list "\$pca_dim" \\
    --batch_size "\$batch_size" \\
    --calib_batches "\$calib_batches" \\
    --logits_batches "\$logits_batches" \\
    --output_dir "\$exp_dir" \\
    --extract_logits \\
    --check_train_structure \\
    --save_csv \\
    --log_file "\$exp_dir/experiment.log" \\
    --verbose

# Check if experiment completed successfully
if [ \$? -eq 0 ]; then
    echo "✅ Experiment completed successfully!"
    echo "COMPLETED: \$(date)" > "\$exp_dir/experiment_completed.txt"
    echo "Results saved in: \$exp_dir"
else
    echo "❌ Experiment failed!"
    echo "FAILED: \$(date)" > "\$exp_dir/experiment_failed.txt"
    echo "Check logs in: \$exp_dir"
fi

echo "------------------------------------------"
echo "🎉 Experiment finished!"
echo "Results directory: \$exp_dir"
EOF

            # Make the file executable
            chmod +x "$filename"
            
            echo "✅ Created: $filename"
            file_count=$((file_count + 1))
        done
    done
done

echo ""
echo "🎉 Generated $file_count combination files!"
echo "=========================================="
echo "Files created:"
echo ""

# List all created files
for adv_mode in "${adv_modes[@]}"; do
    for model in "${models[@]}"; do
        for quant_model in "${quant_models[@]}"; do
            filename="${adv_mode}_${quant_model}_${model}.sh"
            echo "  - $filename"
        done
    done
done

echo ""
echo "🚀 To run any experiment:"
echo "  ./adaround_fixed_resnet18.sh"
echo "  ./brecq_learnable_resnet50.sh"
echo "  ./qdrop_lsq_mnasnet.sh"
echo "  # etc..."
echo ""
echo "📁 Results will be saved in: results/[combination_name]/"
