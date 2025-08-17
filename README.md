# MQBench PTQ Script with Compatibility Fixes

This repository contains a fixed version of the MQBench PTQ script that handles the common `LearnableFakeQuantize` compatibility issue and now supports both training and validation data loaders.

## The Problem

The original script fails with this error:
```
AttributeError: 'LearnableFakeQuantize' object has no attribute 'init'
```

This typically occurs due to version mismatches between:
- MQBench library
- PyTorch version
- Quantizer implementations

## Solutions Provided

### 1. Automatic Fallback
The script now automatically detects compatibility issues and falls back to basic PTQ when advanced PTQ reconstruction fails.

### 2. Command Line Options
- `--no_advanced`: Force disable advanced PTQ reconstruction
- `--diagnose`: Run MQBench setup diagnostics
- `--extract_logits`: Extract and save model logits for analysis
- `--check_train_structure`: Verify ImageNet training data structure

### 3. Better Error Messages
Clear error messages explaining the issue and suggesting solutions.

### 4. Training Data Support
- Creates both training and validation data loaders
- Supports ImageNet directory structure with `train/` and `val/` subfolders
- Optional logits extraction for model analysis

## Directory Structure

The script expects ImageNet data organized as:
```
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ... (1000 class folders)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (1000 class folders)
```

## Usage

### Basic PTQ (Recommended for compatibility issues)
```bash
python mq_bench_ptq.py --model resnet18 --no_advanced
```

### Advanced PTQ (if compatible)
```bash
python mq_bench_ptq.py --model resnet18 --advanced
```

### With Training Data and Logits Extraction
```bash
python mq_bench_ptq.py --model resnet18 --extract_logits --check_train_structure
```

### With PCA Analysis
```bash
python mq_bench_ptq.py --model resnet18 --extract_logits --alpha 0.7 --num_clusters 15 --pca_dim 100
```

### With Multiple PCA Parameters
```bash
python mq_bench_ptq.py --model resnet18 --extract_logits \
    --alpha_list 0.3 0.5 0.7 \
    --num_clusters_list 5 10 15 \
    --pca_dim_list 25 50 100
```

### With Custom CSV Output
```bash
# Save to custom directory
python mq_bench_ptq.py --model resnet18 --extract_logits --output_dir my_results

# Disable CSV export
python mq_bench_ptq.py --model resnet18 --extract_logits --no_save_csv
```

### Run Diagnostics
```bash
python mq_bench_ptq.py --diagnose
```

### Test Compatibility
```bash
python test_compatibility.py
```

## New Command Line Options

- `--val_root`: Path to ImageNet root directory (default: `/home/alz07xz/imagenet`)
- `--extract_logits`: Extract and save model logits for analysis
- `--logits_batches`: Number of batches to use for logits extraction (default: 10)
- `--check_train_structure`: Verify ImageNet training data structure

### PCA Analysis Arguments
- `--alpha`: Alpha parameter for PCA analysis (default: 0.5)
- `--alpha_list`: List of alpha values for PCA analysis (overrides --alpha)
- `--num_clusters`: Number of clusters for PCA analysis (default: 10)
- `--num_clusters_list`: List of cluster numbers for PCA analysis (overrides --num_clusters)
- `--pca_dim`: PCA dimension for analysis (default: 50)
- `--pca_dim_list`: List of PCA dimensions for analysis (overrides --pca_dim)

### Output Arguments
- `--output_dir`: Directory to save CSV results (default: "results")
- `--save_csv`: Save results to CSV files (default: True)

## Troubleshooting

### If you still get the 'init' method error:

1. **Use basic PTQ only:**
   ```bash
   python mq_bench_ptq.py --model resnet18 --no_advanced
   ```

2. **Update MQBench:**
   ```bash
   pip install --upgrade mqbench
   ```

3. **Check PyTorch version compatibility:**
   - MQBench may require specific PyTorch versions
   - Check MQBench documentation for version requirements

4. **Run diagnostics:**
   ```bash
   python mq_bench_ptq.py --diagnose
   ```

## What the Fixes Do

1. **Graceful Degradation**: When advanced PTQ fails, the script continues with basic PTQ instead of crashing.

2. **Better Error Handling**: Clear error messages explaining what went wrong and how to fix it.

3. **Compatibility Checking**: Pre-flight checks to detect issues before they cause crashes.

4. **Fallback Options**: Multiple ways to run the script even when advanced features aren't available.

5. **Training Data Support**: Creates training data loaders for additional analysis capabilities.

6. **Logits Extraction**: Optional extraction and saving of model outputs for further analysis.

7. **PCA Analysis**: Advanced analysis of quantization effects using Principal Component Analysis and clustering techniques.

8. **CSV Export**: Comprehensive logging of all parameters and results for analysis and comparison.

## CSV Output Format

The script generates two CSV files in the specified output directory:

### 1. Detailed Results (`ptq_results_YYYYMMDD_HHMMSS.csv`)
Contains all parameter combinations and their corresponding accuracies:
- **Setup Parameters**: Model, weights, batch size, calibration settings, quantization parameters
- **Advanced PTQ Settings**: Mode, steps, warmup, lambda, probability, GPU usage
- **PCA Parameters**: Alpha, number of clusters, PCA dimensions
- **Results**: Top-1 and Top-5 accuracies for each combination
- **Baseline**: Original PTQ accuracy for comparison

### 2. Summary Results (`ptq_summary_YYYYMMDD_HHMMSS.csv`)
Shows the best result for each unique parameter combination:
- **Best Alpha**: Optimal alpha value for each cluster/PCA combination
- **Improvement**: Accuracy improvement over baseline PTQ
- **Parameter Optimization**: Best settings for each configuration

## Expected Behavior

- **With compatible setup**: Advanced PTQ works normally
- **With incompatible setup**: Script falls back to basic PTQ with warnings
- **With --no_advanced flag**: Skips advanced PTQ entirely
- **With --diagnose flag**: Shows detailed setup information
- **With --extract_logits flag**: Saves model logits for analysis

## Files

- `mq_bench_ptq.py`: Main PTQ script with compatibility fixes and training data support
- `test_compatibility.py`: Standalone compatibility test script
- `README.md`: This documentation

## Notes

- Basic PTQ will still work even when advanced PTQ fails
- Accuracy may be lower without advanced reconstruction techniques
- The script maintains all original functionality when compatible
- All fixes are backward compatible
- Training data is now available for additional analysis
- Logits extraction provides insights into quantization effects
