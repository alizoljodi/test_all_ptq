# SLURM Setup for Large-Scale MQBench PTQ Experiments

This directory contains everything you need to run large-scale PTQ experiments across multiple GPUs using SLURM.

## 🎯 Experiment Overview

**Total Experiments**: 1,920 combinations
**Concurrent Jobs**: 8 (one per GPU)
**Estimated Runtime**: ~24 hours
**Memory per Job**: 32GB RAM
**GPUs per Job**: 1

## 📊 Parameter Combinations

| Parameter | Values | Count |
|-----------|---------|-------|
| **Models** | resnet18, resnet50, mnasnet, mobilenet_v2 | 4 |
| **Advanced Modes** | AdaRound, BRECQ, QDrop | 3 |
| **Weight Bits** | 2, 4, 8 | 3 |
| **Activation Bits** | 2, 4, 8 | 3 |
| **Quantization Models** | fixed, learnable, lsq, lsqplus | 4 |
| **Alpha Values** | 0.2, 0.4, 0.6, 0.8, 1.0 | 5 |
| **Number of Clusters** | 8, 16, 32, 64 | 4 |
| **PCA Dimensions** | 25, 50, 100 | 3 |

**Total**: 4 × 3 × 3 × 3 × 4 × 5 × 4 × 3 = **1,920 experiments**

## 🚀 Quick Start

### 1. Submit Jobs (Simplest)
```bash
chmod +x submit_job.sh
./submit_job.sh
```

### 2. Use Python Manager (Recommended)
```bash
python submit_experiments.py
```

### 3. Manual Submission
```bash
sbatch run_ptq_experiments.slurm
```

## 📁 File Structure

```
├── run_ptq_experiments.slurm    # Main SLURM script
├── submit_experiments.py         # Python experiment manager
├── submit_job.sh                 # Quick submission script
├── mq_bench_ptq.py              # Your PTQ script
├── logs/                         # SLURM output logs
├── results/                      # Experiment results
└── SLURM_README.md              # This file
```

## ⚙️ SLURM Configuration

### Resource Requirements
- **Nodes**: 1
- **Tasks**: 1 per job
- **CPUs per Task**: 8
- **Memory**: 32GB per job
- **GPUs**: 1 per job
- **Time Limit**: 24 hours

### Array Configuration
- **Array Range**: 0-1919 (1920 total jobs)
- **Concurrency**: 8 jobs max (one per GPU)
- **Format**: `--array=0-1919%8`

## 🔧 Customization

### Modify Parameters
Edit the parameter arrays in `run_ptq_experiments.slurm`:

```bash
# Change models
models=("resnet18" "resnet50" "mnasnet" "mobilenet_v2")

# Change bit widths
w_bits=(2 4 8)
a_bits=(2 4 8)

# Change advanced modes
adv_modes=("adaround" "brecq" "qdrop")
```

### Adjust Resources
Modify the SLURM headers in `run_ptq_experiments.slurm`:

```bash
#SBATCH --mem=64G              # Increase memory
#SBATCH --time=48:00:00        # Increase time limit
#SBATCH --array=0-1919%16      # Increase concurrency to 16
```

### Change GPU Count
Update the array concurrency:
```bash
#SBATCH --array=0-1919%8       # 8 concurrent jobs
#SBATCH --array=0-1919%16      # 16 concurrent jobs
#SBATCH --array=0-1919%32      # 32 concurrent jobs
```

## 📊 Monitoring and Control

### Check Job Status
```bash
# Check specific job
squeue -j <JOB_ID>

# Check all your jobs
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>
```

### View Logs
```bash
# View SLURM output
tail -f logs/ptq_<JOB_ID>_<ARRAY_ID>.out

# View error logs
tail -f logs/ptq_<JOB_ID>_<ARRAY_ID>.err

# View experiment logs
tail -f results/<EXPERIMENT_NAME>_<TIMESTAMP>/<EXPERIMENT_NAME>_<TIMESTAMP>.log
```

### Cancel Jobs
```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel specific array job
scancel <JOB_ID>_<ARRAY_ID>
```

## 📈 Results and Analysis

### Output Structure
Each experiment creates:
```
results/
├── <MODEL>_<ADV_MODE>_w<WBITS>a<ABITS>_<QUANT_MODEL>_a<ALPHA>_c<CLUSTERS>_pca<PCA_DIM>_<TIMESTAMP>/
│   ├── ptq_results_<TIMESTAMP>.csv          # Detailed results
│   ├── ptq_summary_<TIMESTAMP>.csv          # Summary results
│   ├── experiment_completed.txt              # Completion marker
│   └── <EXPERIMENT_NAME>_<TIMESTAMP>.log    # Experiment log
```

### CSV Files
- **Detailed Results**: All parameter combinations with accuracies
- **Summary Results**: Best results for each parameter combination
- **Improvement Metrics**: Accuracy improvement over baseline PTQ

### Analysis Scripts
Use the Python manager to:
- Generate summary reports
- Check completion status
- Monitor experiment progress
- Analyze results

## 🚨 Troubleshooting

### Common Issues

#### 1. Memory Errors
**Problem**: CUDA out of memory
**Solution**: Reduce batch size or calibration batches in the SLURM script

#### 2. Time Limit Exceeded
**Problem**: Jobs killed due to time limit
**Solution**: Increase `--time` in SLURM headers

#### 3. GPU Not Available
**Problem**: No GPUs available
**Solution**: Check GPU availability with `sinfo` and adjust queue

#### 4. Module Not Found
**Problem**: Python/CUDA modules not found
**Solution**: Update module paths in the SLURM script

### Debug Commands
```bash
# Check SLURM configuration
sinfo -N -l

# Check available partitions
sinfo -s

# Check queue status
squeue -p <PARTITION_NAME>

# Check job history
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize
```

## 📋 Best Practices

### 1. Resource Management
- Start with smaller concurrency (8 jobs)
- Monitor memory usage and adjust accordingly
- Use appropriate time limits for your models

### 2. Data Organization
- Use descriptive experiment names
- Organize results by timestamp
- Keep logs for debugging

### 3. Monitoring
- Check job status regularly
- Monitor resource usage
- Set up email notifications

### 4. Backup
- Save experiment configurations
- Backup important results
- Document parameter changes

## 🔄 Scaling Up

### Increase Concurrency
```bash
# Change from 8 to 16 concurrent jobs
#SBATCH --array=0-1919%16
```

### Multi-Node Setup
```bash
# Use multiple nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
```

### Different Models
Add more models to the arrays:
```bash
models=("resnet18" "resnet50" "mnasnet" "mobilenet_v2" "vit_b_16" "efficientnet_b0")
```

## 📞 Support

### SLURM Commands
- `man sbatch` - SLURM submission manual
- `man squeue` - Queue monitoring manual
- `man scontrol` - Job control manual

### Useful Links
- [SLURM Documentation](https://slurm.schedmd.com/)
- [SLURM Quick Start](https://slurm.schedmd.com/quickstart.html)
- [SLURM Array Jobs](https://slurm.schedmd.com/job_array.html)

### Contact
For issues with:
- **SLURM**: Contact your cluster administrator
- **MQBench**: Check MQBench documentation
- **Scripts**: Check the logs and error messages

---

**Happy Experimenting! 🚀**
