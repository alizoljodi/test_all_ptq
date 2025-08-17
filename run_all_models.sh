#!/bin/bash

# Master script to run all model PTQ experiments
# Runs each model sequentially

echo "🚀 Starting ALL Model PTQ Experiments"
echo "=========================================="
echo "Models to run:"
echo "  1. ResNet18"
echo "  2. ResNet50" 
echo "  3. MNASNet"
echo "  4. MobileNet V2"
echo "=========================================="

# Create main results directory
mkdir -p results
cd results

# Run ResNet18
echo ""
echo "🔄 Starting ResNet18 experiments..."
echo "=========================================="
cd ..
bash run_resnet18.sh
if [ $? -eq 0 ]; then
    echo "✅ ResNet18 experiments completed successfully"
else
    echo "❌ ResNet18 experiments failed"
    exit 1
fi

# Run ResNet50
echo ""
echo "🔄 Starting ResNet50 experiments..."
echo "=========================================="
bash run_resnet50.sh
if [ $? -eq 0 ]; then
    echo "✅ ResNet50 experiments completed successfully"
else
    echo "❌ ResNet50 experiments failed"
    exit 1
fi

# Run MNASNet
echo ""
echo "🔄 Starting MNASNet experiments..."
echo "=========================================="
bash run_mnasnet.sh
if [ $? -eq 0 ]; then
    echo "✅ MNASNet experiments completed successfully"
else
    echo "❌ MNASNet experiments failed"
    exit 1
fi

# Run MobileNet V2
echo ""
echo "🔄 Starting MobileNet V2 experiments..."
echo "=========================================="
bash run_mobilenet_v2.sh
if [ $? -eq 0 ]; then
    echo "✅ MobileNet V2 experiments completed successfully"
else
    echo "❌ MobileNet V2 experiments failed"
    exit 1
fi

echo ""
echo "🎉 ALL MODEL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Results saved in: results/"
echo "  - results/resnet18/"
echo "  - results/resnet50/"
echo "  - results/mnasnet/"
echo "  - results/mobilenet_v2/"
echo ""
echo "Total experiments per model: 60 (5×4×3)"
echo "Total experiments across all models: 240"
echo "=========================================="
