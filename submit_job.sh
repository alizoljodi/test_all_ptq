#!/bin/bash
# Quick SLURM job submission script with configurable concurrency

# Default concurrency
MAX_CONCURRENT=${1:-8}

echo "🚀 Submitting MQBench PTQ Experiments to SLURM..."
echo "⚙️  Max Concurrent Jobs: $MAX_CONCURRENT"
echo "📊 Total Experiments: 1,920"
echo "🎯 Job Name: mqbench_ptq_experiments"

# Check if sbatch is available
if ! command -v sbatch &> /dev/null; then
    echo "❌ Error: sbatch command not found. Are you on a SLURM cluster?"
    exit 1
fi

# Check if the SLURM script exists
if [ ! -f "run_ptq_experiments.slurm" ]; then
    echo "❌ Error: run_ptq_experiments.slurm not found!"
    exit 1
fi

# Create logs and results directories
mkdir -p logs results

# Submit the job with configurable concurrency
echo "📤 Submitting SLURM array job..."
job_output=$(python submit_experiments.py --max-concurrent $MAX_CONCURRENT 2>&1)

if [ $? -eq 0 ]; then
    echo "✅ Job submitted successfully!"
    echo ""
    echo "📋 Job Details:"
    echo "   Job Name: mqbench_ptq_experiments"
    echo "   Output Logs: logs/mqbench_ptq_*.out"
    echo "   Error Logs: logs/mqbench_ptq_*.err"
    echo "   Results: results/"
    echo ""
    echo "📊 Use the following commands to monitor your job:"
    echo "   squeue -u $USER                    # Check job status"
    echo "   squeue -n mqbench_ptq_experiments # Check specific job"
    echo "   tail -f logs/mqbench_ptq_*.out    # Monitor output logs"
    echo "   tail -f logs/mqbench_ptq_*.err    # Monitor error logs"
else
    echo "❌ Failed to submit job:"
    echo "$job_output"
    exit 1
fi
