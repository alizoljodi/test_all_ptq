#!/bin/bash

# Script to list all combinations and show which files exist

echo "📋 All Required Combination Files"
echo "=========================================="

# Define parameter lists
adv_modes=("adaround" "brecq" "qdrop")
models=("resnet18" "resnet50" "mnasnet" "mobilenet_v2")
quant_models=("fixed" "learnable" "lsq" "lsqplus")

# Counter
total=0
existing=0
missing=0

echo "Required files (48 total):"
echo ""

for adv_mode in "${adv_modes[@]}"; do
    for model in "${models[@]}"; do
        for quant_model in "${quant_models[@]}"; do
            filename="${adv_mode}_${quant_model}_${model}.sh"
            total=$((total + 1))
            
            if [ -f "$filename" ]; then
                echo "✅ $filename"
                existing=$((existing + 1))
            else
                echo "❌ $filename (MISSING)"
                missing=$((missing + 1))
            fi
        done
    done
    echo ""
done

echo "=========================================="
echo "📊 Summary:"
echo "  Total required: $total"
echo "  ✅ Existing: $existing"
echo "  ❌ Missing: $missing"
echo ""

if [ $missing -gt 0 ]; then
    echo "🔧 To create missing files, run:"
    echo "  python create_all_files.py"
    echo ""
    echo "Or manually create the missing files following the pattern:"
    echo "  [adv_mode]_[quant_model]_[model].sh"
fi
