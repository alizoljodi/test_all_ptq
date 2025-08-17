#!/bin/bash
#SBATCH -J TestJob
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:4
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>

source mqbench/bin/activate


echo "🚀 Submitting MQBench PTQ Experiments to SLURM..."
echo "=========================================="

# Check if SLURM is available
if ! command -v sbatch &> /dev/null; then
    echo "❌ Error: SLURM (sbatch) not found. Are you on a SLURM cluster?"
    exit 1
fi

# Check if the SLURM script exists
if [ ! -f "run_ptq_experiments.slurm" ]; then
    echo "❌ Error: run_ptq_experiments.slurm not found!"
    exit 1
fi

# Create necessary directories
mkdir -p logs
mkdir -p results

echo "📁 Created logs/ and results/ directories"

# Submit the job
echo "📤 Submitting SLURM array job..."
job_output=$(sbatch run_ptq_experiments.slurm 2>&1)

if [ $? -eq 0 ]; then
    # Extract job ID
    job_id=$(echo "$job_output" | grep -o '[0-9]\+' | head -1)
    echo "✅ Job submitted successfully!"
    echo "📊 Job ID: $job_id"
    echo "🔢 Total experiments: 1,920"
    echo "🚀 Max concurrent jobs: 8"
    echo "⏱️  Estimated runtime: ~24 hours"
    echo ""
    echo "💡 Useful commands:"
    echo "   Check job status: squeue -j $job_id"
    echo "   Monitor all jobs: squeue -u \$USER"
    echo "   Cancel job: scancel $job_id"
    echo "   View logs: tail -f logs/ptq_${job_id}_*.out"
    echo ""
    echo "📁 Results will be saved to: results/"
    echo "📝 Logs will be saved to: logs/"
    
else
    echo "❌ Failed to submit job:"
    echo "$job_output"
    exit 1
fi
