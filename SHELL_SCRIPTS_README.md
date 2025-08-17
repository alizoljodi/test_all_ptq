# Simple Shell Scripts for MQBench PTQ Experiments

This directory contains simple shell scripts to run `mq_bench_ptq.py` with different parameter combinations.

## 📁 Available Scripts

### Basic Model Scripts (PCA Parameters Only)
- **`run_resnet18.sh`** - Run ResNet18 experiments (60 experiments)
- **`run_resnet50.sh`** - Run ResNet50 experiments (60 experiments)  
- **`run_mnasnet.sh`** - Run MNASNet experiments (60 experiments)
- **`run_mobilenet_v2.sh`** - Run MobileNet V2 experiments (60 experiments)

### Advanced Quantization Scripts
- **`run_resnet18_advanced.sh`** - Run ResNet18 with all quantization options (2,160 experiments)
- **`run_comprehensive_comparison.sh`** - Run all models with all quantization options (8,640 experiments)
- **`run_quantization_analysis.sh`** - Focused analysis with fixed PCA parameters (288 experiments)

### Master Script
- **`run_all_models.sh`** - Run basic experiments for all models sequentially (240 experiments)

## 🚀 How to Use

### 1. Make Scripts Executable
```bash
chmod +x run_*.sh
```

### 2. Run Basic Experiments (PCA Parameters Only)
```bash
# Run just ResNet18 (60 experiments)
./run_resnet18.sh

# Run all models sequentially (240 experiments)
./run_all_models.sh
```

### 3. Run Advanced Quantization Experiments
```bash
# Run ResNet18 with all quantization options (2,160 experiments)
./run_resnet18_advanced.sh

# Run comprehensive comparison across all models (8,640 experiments)
./run_comprehensive_comparison.sh

# Run focused quantization analysis (288 experiments)
./run_quantization_analysis.sh
```

## 📊 Parameter Combinations

### Basic Scripts (60 experiments per model)
- **Alpha values**: 0.2, 0.4, 0.6, 0.8, 1.0 (5 values)
- **Number of clusters**: 8, 16, 32, 64 (4 values)  
- **PCA dimensions**: 25, 50, 100 (3 values)

**Total**: 5 × 4 × 3 = 60 experiments per model

### Advanced Scripts (Full Quantization Analysis)
- **Models**: resnet18, resnet50, mnasnet, mobilenet_v2 (4 models)
- **Quant Models**: fixed, learnable, lsq, lsqplus (4 methods)
- **Advanced Modes**: adaround, brecq, qdrop (3 techniques)
- **Weight Bits**: 2, 4, 8 (3 bit widths)
- **Activation Bits**: 2, 4, 8 (3 bit widths)
- **Alpha values**: 0.2, 0.4, 0.6, 0.8, 1.0 (5 values)
- **Number of clusters**: 8, 16, 32, 64 (4 values)
- **PCA dimensions**: 25, 50, 100 (3 values)

**Total Comprehensive**: 4 × 4 × 3 × 3 × 3 × 5 × 4 × 3 = 8,640 experiments

## 🔬 Advanced Quantization Features

### Quantization Models
- **`fixed`**: Standard fixed-point quantization
- **`learnable`**: Learnable quantization parameters
- **`lsq`**: Learned Step Size Quantization
- **`lsqplus`**: Enhanced LSQ with additional features

### Advanced PTQ Techniques
- **`adaround`**: AdaRound for weight rounding optimization
- **`brecq`**: BRECQ for block reconstruction
- **`qdrop`**: QDrop for activation quantization

### Bit Width Analysis
- **2-bit**: Ultra-low precision (highest compression, potential accuracy loss)
- **4-bit**: Low precision (good compression, moderate accuracy)
- **8-bit**: Standard precision (balanced compression and accuracy)

## 📁 Output Structure

### Basic Scripts
```
results/
├── resnet18/
│   ├── exp_1_alpha0.2_clusters8_pca25/
│   ├── exp_2_alpha0.2_clusters8_pca50/
│   └── ... (60 total)
└── resnet50/
    └── ... (60 total)
```

### Advanced Scripts
```
results/
├── resnet18_advanced/
│   ├── exp_1_fixed_adaround_w2a2_alpha0.2_clusters8_pca25/
│   ├── exp_2_fixed_adaround_w2a2_alpha0.2_clusters8_pca50/
│   └── ... (2,160 total)
├── comprehensive_comparison/
│   ├── exp_1_resnet18_fixed_adaround_w2a2_alpha0.2_clusters8_pca25/
│   └── ... (8,640 total)
└── quantization_analysis/
    ├── exp_1_resnet18_fixed_adaround_w2a2_fixedPCA/
    └── ... (288 total)
```

## ⚙️ Model-Specific Settings

### ResNet18 & MNASNet & MobileNet V2
- Batch size: 64 (8-bit), 48 (4-bit), 32 (2-bit)
- Calibration batches: 32 (8-bit), 24 (4-bit), 16 (2-bit)
- Logits batches: 10 (8-bit), 8 (4-bit), 5 (2-bit)

### ResNet50 (larger model)
- Batch size: 48 (8-bit), 36 (4-bit), 24 (2-bit)
- Calibration batches: 24 (8-bit), 18 (4-bit), 12 (2-bit)
- Logits batches: 8 (8-bit), 6 (4-bit), 4 (2-bit)

## 🔍 Monitoring Progress

Each script shows:
- Current experiment number and total
- All parameters being tested
- Timestamp for each experiment
- Success/failure status
- Progress through all combinations

## 💾 Results Tracking

- **`experiment_completed.txt`** - Created when experiment succeeds
- **`experiment_failed.txt`** - Created when experiment fails
- **`experiment.log`** - Detailed log for each experiment
- **CSV results** - Saved in each experiment directory

## ⏱️ Expected Runtime

### Basic Scripts
- **Per model**: ~2-4 hours (depending on GPU)
- **All models**: ~8-16 hours total
- **Individual experiment**: ~2-4 minutes

### Advanced Scripts
- **ResNet18 Advanced**: ~72-144 hours (3-6 days)
- **Comprehensive Comparison**: ~288-576 hours (12-24 days)
- **Quantization Analysis**: ~9-18 hours

## 🚨 Troubleshooting

### If a script fails:
1. Check the error message in the terminal
2. Look for `experiment_failed.txt` files
3. Check individual experiment logs
4. Resume manually by running the specific script again

### Memory issues:
- Reduce `--batch_size` in the script
- Reduce `--calib_batches` 
- Reduce `--logits_batches`

### For very long experiments:
- Consider running individual model scripts separately
- Use `run_quantization_analysis.sh` for focused analysis
- Monitor system resources during long runs

## 📝 Customization

To modify parameters, edit the arrays in each script:

```bash
# Change quantization models
quant_models=("fixed" "learnable")

# Change advanced modes
adv_modes=("adaround" "brecq")

# Change bit widths
w_bits=(4 8)
a_bits=(4 8)

# Change PCA parameters
alphas=(0.3 0.5 0.7)
num_clusters=(8 16 32)
pca_dims=(25 50)
```

## 🎯 Quick Start Recommendations

### For Beginners
```bash
# Start with basic experiments
./run_resnet18.sh
```

### For Quantization Research
```bash
# Run focused analysis (manageable size)
./run_quantization_analysis.sh
```

### For Comprehensive Studies
```bash
# Run full analysis (very long, but complete)
./run_comprehensive_comparison.sh
```

### For Specific Model Analysis
```bash
# Run advanced experiments for specific model
./run_resnet18_advanced.sh
```

## 🔬 Research Questions Answered

These scripts help answer:
1. **Quantization Impact**: How do different quantization methods affect accuracy?
2. **Bit Width Trade-offs**: What's the accuracy vs compression trade-off?
3. **Advanced PTQ**: Which reconstruction technique works best?
4. **Model Sensitivity**: How do different architectures respond to quantization?
5. **PCA Clustering**: What's the optimal clustering strategy for recovery?

That's it! Choose the script that matches your research needs. 🎉
