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
- `--recover`: Recover from existing results and resume incomplete experiments

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

### Crash Recovery and Resume
```bash
# Resume from where you left off after a crash
python mq_bench_ptq.py --model resnet18 --extract_logits --recover

# Resume with custom output directory
python mq_bench_ptq.py --model resnet18 --extract_logits --recover --output_dir my_results
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

9. **Accuracy Tracking**: Complete accuracy comparison including FP32 baseline, PTQ baseline, and clustering recovery.

## Crash Recovery and Resume

The script now includes robust crash recovery functionality that allows you to resume experiments from where they left off.

### How It Works

1. **Automatic Checkpointing**: The script saves progress every 5 combinations and creates recovery checkpoints
2. **Result Persistence**: All completed results are saved to CSV files with timestamps
3. **Smart Resume**: When restarting with `--recover`, the script automatically detects completed combinations
4. **Progress Tracking**: Shows exactly how many combinations remain to be completed

### Recovery Files

- **`ptq_results_YYYYMMDD_HHMMSS.csv`**: Detailed results with timestamps
- **`recovery_checkpoint.json`**: Progress tracking and experiment state
- **`ptq_summary_YYYYMMDD_HHMMSS.csv`**: Summary of best results

### Usage Examples

#### Resume After Crash
```bash
# If your experiment crashes, simply restart with --recover
python mq_bench_ptq.py --model resnet18 --extract_logits --recover
```

#### Resume After Manual Interruption
```bash
# If you stop with Ctrl+C, restart with --recover
python mq_bench_ptq.py --model resnet18 --extract_logits --recover
```

#### Check Recovery Status
```bash
# The script will show you exactly what's been completed and what remains
python mq_bench_ptq.py --model resnet18 --extract_logits --recover --verbose
```

### What Gets Recovered

- ✅ **Completed combinations**: All previously run parameter combinations
- ✅ **Accuracy results**: Top-1 and Top-5 accuracies for each combination
- ✅ **Experiment parameters**: Model, quantization settings, PCA parameters
- ✅ **Progress state**: Exact point where the experiment stopped

### What Gets Skipped

- ❌ **Already completed combinations**: Won't re-run experiments that are done
- ❌ **Duplicate work**: Efficiently resumes from the last saved state
- ❌ **Data loss**: All progress is preserved in CSV and checkpoint files

### Recovery Scenarios

1. **System Crash**: Restart with `--recover` to continue from last checkpoint
2. **Power Outage**: All progress is saved, resume seamlessly
3. **Manual Stop**: Ctrl+C interruption saves current state
4. **SLURM Timeout**: Job can be resubmitted with `--recover`
5. **Memory Errors**: Fix memory issues and resume from last successful point

## CSV Output Format

The script generates two CSV files in the specified output directory:

### 1. Detailed Results (`ptq_results_YYYYMMDD_HHMMSS.csv`)
Contains all parameter combinations and their corresponding accuracies:
- **Setup Parameters**: Model, weights, batch size, calibration settings, quantization parameters
- **Advanced PTQ Settings**: Mode, steps, warmup, lambda, probability, GPU usage
- **Baseline Accuracies**: FP32 model accuracy and baseline PTQ accuracy
- **PCA Parameters**: Alpha, number of clusters, PCA dimensions
- **Results**: Top-1 and Top-5 accuracies for each combination
- **Comparison Metrics**: PTQ degradation and clustering recovery

### 2. Summary Results (`ptq_summary_YYYYMMDD_HHMMSS.csv`)
Shows the best result for each unique parameter combination:
- **Best Alpha**: Optimal alpha value for each cluster/PCA combination
- **Improvement Metrics**: Accuracy improvement over baseline PTQ and FP32
- **Parameter Optimization**: Best settings for each configuration
- **Recovery Analysis**: How much accuracy was recovered from clustering

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

## SLURM Automation

The repository includes a complete SLURM setup to automate large-scale PTQ experiments:

### Quick Start with SLURM

#### Option 1: Python Script (Recommended)
```bash
# Run with default 8 concurrent jobs
python submit_experiments.py

# Run with custom concurrency (e.g., 4 concurrent jobs)
python submit_experiments.py --max-concurrent 4

# Run with maximum concurrency (e.g., 16 concurrent jobs)
python submit_experiments.py --max-concurrent 16
```

#### Option 2: Shell Script
```bash
# Run with default 8 concurrent jobs
./submit_job.sh

# Run with custom concurrency (e.g., 4 concurrent jobs)
./submit_job.sh 4

# Run with maximum concurrency (e.g., 16 concurrent jobs)
./submit_job.sh 16
```

#### Option 3: Direct SLURM Submission
```bash
# Submit directly to SLURM (modify the script first if you want different concurrency)
sbatch run_ptq_experiments.slurm
```

### SLURM Job Details

- **Job Name**: `mqbench_ptq_experiments`
- **Output Logs**: `logs/mqbench_ptq_<JOB_ID>_<ARRAY_ID>.out`
- **Error Logs**: `logs/mqbench_ptq_<JOB_ID>_<ARRAY_ID>.err`
- **Results Directory**: `results/`
- **Temporary Scripts**: `temp_slurm_concurrent<N>_<TIMESTAMP>.sh`

### Monitoring Commands

```bash
# Check all your jobs
squeue -u $USER

# Check specific job by name
squeue -n mqbench_ptq_experiments

# Monitor output logs
tail -f logs/mqbench_ptq_*.out

# Monitor error logs
tail -f logs/mqbench_ptq_*.err

# Check specific array job
tail -f logs/mqbench_ptq_<JOB_ID>_<ARRAY_ID>.out
```

### Configurable Concurrency

The `--max-concurrent` parameter controls how many **sequential batches** run:

- **`--max-concurrent 2`**: 2 batches run one after another (very conservative)
- **`--max-concurrent 4`**: 4 batches run sequentially (default, balanced)
- **`--max-concurrent 8`**: 8 batches run sequentially (for large clusters)

**Total Core Experiments**: 96 combinations (4 models × 3 adv_modes × 3 w_bits × 3 a_bits × 4 quant_models)
**PCA Combinations per Core**: 60 combinations (5 alphas × 4 clusters × 3 PCA dims)
**Total Parameter Tests**: 5,760 combinations (96 × 60)
**Batch Size**: ~24 core experiments per batch (with 4 batches)
**Execution**: **Sequential** - each batch waits for the previous one to complete

### How It Works

Instead of one large array job that violates cluster policies, the system now:

1. **Divides core experiments into batches**: 96 core experiments → 4 batches of ~24 each
2. **Each core experiment tests all PCA parameters**: Every core experiment runs with all 60 PCA combinations
3. **Submits batches sequentially**: Batch 1 → wait → Batch 2 → wait → Batch 3 → wait → Batch 4
4. **Avoids policy violations**: Each batch is a separate job with reasonable resource requests
5. **Maintains progress tracking**: Each batch saves results independently

### Resource Requirements

Each individual batch job requires:
- **1 GPU** (per batch, not per experiment)
- **4 CPUs** (conservative)
- **16GB RAM** (conservative)
- **12 hours time limit** (per batch)

### Execution Timeline

With 4 batches:
- **Batch 1**: Hours 0-12 (core experiments 0-23, each testing 60 PCA combinations)
- **Batch 2**: Hours 12-24 (core experiments 24-47, each testing 60 PCA combinations)
- **Batch 3**: Hours 24-36 (core experiments 48-71, each testing 60 PCA combinations)
- **Batch 4**: Hours 36-48 (core experiments 72-95, each testing 60 PCA combinations)

**Total Runtime**: ~48 hours (sequential execution)
**Total Parameter Tests**: 5,760 combinations across all batches

### Troubleshooting Cluster Limits

If you get `QOSMaxGRESPerJob` or similar errors:

1. **Reduce concurrency**:
   ```bash
   python submit_experiments.py --max-concurrent 2
   ```

2. **Check cluster limits**:
   ```bash
   sinfo --format "%P %G %m %c %f %D %t"
   squeue -u $USER
   ```

3. **Use smaller resource requests**:
   - Modify `run_ptq_experiments.slurm` to reduce CPU/memory requirements
   - Contact your cluster administrator for proper resource limits
