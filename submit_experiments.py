#!/usr/bin/env python3
"""
SLURM Experiment Manager for MQBench PTQ Experiments

This script helps submit, monitor, and manage the large-scale PTQ experiments.
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import re

class ExperimentManager:
    def __init__(self, max_concurrent=8):
        self.experiments = []
        self.job_ids = []
        self.max_concurrent = max_concurrent
        self.results_dir = Path("results")
        self.logs_dir = Path("logs")
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
    def generate_experiment_list(self):
        """Generate all experiment combinations."""
        models = ["resnet18", "resnet50", "mnasnet", "mobilenet_v2"]
        adv_modes = ["adaround", "brecq", "qdrop"]
        w_bits = [2, 4, 8]
        a_bits = [2, 4, 8]
        quant_models = ["fixed", "learnable", "lsq", "lsqplus"]
        alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
        num_clusters = [8, 16, 32, 64]
        pca_dims = [25, 50, 100]
        
        experiment_id = 0
        for model in models:
            for adv_mode in adv_modes:
                for w_bit in w_bits:
                    for a_bit in a_bits:
                        for quant_model in quant_models:
                            for alpha in alphas:
                                for num_cluster in num_clusters:
                                    for pca_dim in pca_dims:
                                        exp = {
                                            'id': experiment_id,
                                            'model': model,
                                            'adv_mode': adv_mode,
                                            'w_bits': w_bit,
                                            'a_bits': a_bit,
                                            'quant_model': quant_model,
                                            'alpha': alpha,
                                            'num_clusters': num_cluster,
                                            'pca_dim': pca_dim,
                                            'status': 'pending'
                                        }
                                        self.experiments.append(exp)
                                        experiment_id += 1
        
        print(f"Generated {len(self.experiments)} experiment combinations")
        return self.experiments
    
    def save_experiment_list(self, filename="experiment_list.json"):
        """Save experiment list to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        print(f"Experiment list saved to {filename}")
    
    def submit_slurm_job(self):
        """Submit the SLURM array job with configurable concurrency."""
        try:
            # Create a temporary SLURM script with configurable concurrency
            slurm_script = self._create_slurm_script()
            
            # Submit the SLURM job
            cmd = ["sbatch", slurm_script]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Extract job ID from output
            output = result.stdout.strip()
            if "Submitted batch job" in output:
                job_id = output.split()[-1]
                self.job_ids.append(job_id)
                print(f"✅ SLURM job submitted successfully! Job ID: {job_id}")
                print(f"📊 Total experiments: {len(self.experiments)}")
                print(f"🚀 Max concurrent jobs: {self.max_concurrent}")
                print(f"⏱️  Estimated runtime: ~24 hours")
                
                # Clean up temporary script
                os.remove(slurm_script)
                return job_id
            else:
                print(f"❌ Failed to submit job: {output}")
                # Clean up temporary script
                os.remove(slurm_script)
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error submitting SLURM job: {e}")
            print(f"Error output: {e.stderr}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def _create_slurm_script(self):
        """Create a temporary SLURM script with configurable concurrency."""
        # Read the base SLURM script
        base_script_path = "run_ptq_experiments.slurm"
        if not os.path.exists(base_script_path):
            raise FileNotFoundError(f"Base SLURM script not found: {base_script_path}")
        
        with open(base_script_path, 'r') as f:
            base_content = f.read()
        
        # Replace the concurrency setting
        # Find the line with --array=0-1919%8 and replace the number after %
        pattern = r'--array=0-1919%\d+'
        replacement = f'--array=0-1919%{self.max_concurrent}'
        modified_content = re.sub(pattern, replacement, base_content)
        
        # Create temporary script with better naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_script = f"temp_slurm_concurrent{self.max_concurrent}_{timestamp}.sh"
        with open(temp_script, 'w') as f:
            f.write(modified_content)
        
        return temp_script
    
    def monitor_jobs(self, job_id):
        """Monitor the status of submitted jobs."""
        print(f"🔍 Monitoring job {job_id}...")
        
        while True:
            try:
                # Check job status
                cmd = ["squeue", "-j", job_id, "--format", "%j %t %M %L"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                if result.stdout.strip():
                    # Job is still running
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:  # Skip header
                        status_line = lines[1]
                        parts = status_line.split()
                        if len(parts) >= 4:
                            job_name, status, time_used, time_limit = parts[0], parts[1], parts[2], parts[3]
                            print(f"📊 Job {job_id}: {status} | Time: {time_used}/{time_limit}")
                else:
                    # Job completed
                    print(f"✅ Job {job_id} completed!")
                    break
                
                time.sleep(60)  # Check every minute
                
            except subprocess.CalledProcessError:
                print(f"⚠️  Could not check job status")
                break
            except KeyboardInterrupt:
                print(f"\n⏹️  Monitoring stopped by user")
                break
    
    def check_results(self):
        """Check which experiments have completed."""
        print("🔍 Checking experiment results...")
        
        completed = 0
        failed = 0
        pending = 0
        
        for exp in self.experiments:
            exp_name = f"{exp['model']}_{exp['adv_mode']}_w{exp['w_bits']}a{exp['a_bits']}_{exp['quant_model']}_a{exp['alpha']}_c{exp['num_clusters']}_pca{exp['pca_dim']}"
            
            # Look for results in the results directory
            exp_dirs = list(self.results_dir.glob(f"{exp_name}_*"))
            
            if exp_dirs:
                # Check if experiment completed successfully
                latest_dir = max(exp_dirs, key=lambda x: x.stat().st_mtime)
                completion_file = latest_dir / "experiment_completed.txt"
                failure_file = latest_dir / "experiment_failed.txt"
                
                if completion_file.exists():
                    exp['status'] = 'completed'
                    exp['result_dir'] = str(latest_dir)
                    completed += 1
                elif failure_file.exists():
                    exp['status'] = 'failed'
                    exp['result_dir'] = str(latest_dir)
                    failed += 1
                else:
                    exp['status'] = 'running'
                    pending += 1
            else:
                exp['status'] = 'pending'
                pending += 1
        
        print(f"📊 Experiment Status:")
        print(f"   ✅ Completed: {completed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   🔄 Running: {pending}")
        print(f"   ⏳ Pending: {len(self.experiments) - completed - failed - pending}")
        
        return completed, failed, pending
    
    def generate_summary_report(self):
        """Generate a summary report of all experiments."""
        print("📋 Generating summary report...")
        
        # Check results first
        completed, failed, pending = self.check_results()
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.experiments),
            'completed': completed,
            'failed': failed,
            'pending': pending,
            'completion_rate': f"{(completed / len(self.experiments) * 100):.1f}%",
            'experiments': self.experiments
        }
        
        # Save summary
        summary_file = "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary report saved to {summary_file}")
        
        # Print top-level summary
        print(f"\n🎯 EXPERIMENT SUMMARY")
        print(f"   Total: {len(self.experiments)}")
        print(f"   Completed: {completed} ({(completed/len(self.experiments)*100):.1f}%)")
        print(f"   Failed: {failed} ({(failed/len(self.experiments)*100):.1f}%)")
        print(f"   Pending: {pending} ({(pending/len(self.experiments)*100):.1f}%)")
        
        return summary
    
    def show_experiment_details(self, experiment_id):
        """Show details of a specific experiment."""
        if 0 <= experiment_id < len(self.experiments):
            exp = self.experiments[experiment_id]
            print(f"\n🔬 Experiment {experiment_id} Details:")
            print(f"   Model: {exp['model']}")
            print(f"   Advanced Mode: {exp['adv_mode']}")
            print(f"   Weight Bits: {exp['w_bits']}")
            print(f"   Activation Bits: {exp['a_bits']}")
            print(f"   Quantization Model: {exp['quant_model']}")
            print(f"   Alpha: {exp['alpha']}")
            print(f"   Number of Clusters: {exp['num_clusters']}")
            print(f"   PCA Dimension: {exp['pca_dim']}")
            print(f"   Status: {exp['status']}")
            if 'result_dir' in exp:
                print(f"   Result Directory: {exp['result_dir']}")
        else:
            print(f"❌ Invalid experiment ID: {experiment_id}")

def main():
    """Main function to run the experiment manager."""
    parser = argparse.ArgumentParser(description="MQBench PTQ Experiment Manager")
    parser.add_argument("--max-concurrent", type=int, default=8, 
                       help="Maximum number of concurrent SLURM jobs (default: 8)")
    args = parser.parse_args()

    manager = ExperimentManager(max_concurrent=args.max_concurrent)
    
    print("🚀 MQBench PTQ Experiment Manager")
    print("=" * 50)
    print(f"⚙️  Configuration:")
    print(f"   Max Concurrent Jobs: {args.max_concurrent}")
    print(f"   Total Experiments: 1,920")
    print(f"   Estimated Runtime: ~24 hours")
    print("=" * 50)
    
    # Generate experiment list
    manager.generate_experiment_list()
    manager.save_experiment_list()
    
    while True:
        print(f"\n📋 Available Commands:")
        print(f"   1. Submit SLURM job")
        print(f"   2. Monitor jobs")
        print(f"   3. Check results")
        print(f"   4. Generate summary report")
        print(f"   5. Show experiment details")
        print(f"   6. Exit")
        
        choice = input(f"\n🎯 Enter your choice (1-6): ").strip()
        
        if choice == '1':
            job_id = manager.submit_slurm_job()
            if job_id:
                print(f"💡 Use 'squeue -u $USER' to see all your jobs")
                print(f"💡 Use 'scancel {job_id}' to cancel the job")
        
        elif choice == '2':
            if manager.job_ids:
                for job_id in manager.job_ids:
                    manager.monitor_jobs(job_id)
            else:
                print("❌ No jobs submitted yet. Use option 1 first.")
        
        elif choice == '3':
            manager.check_results()
        
        elif choice == '4':
            manager.generate_summary_report()
        
        elif choice == '5':
            try:
                exp_id = int(input("Enter experiment ID: "))
                manager.show_experiment_details(exp_id)
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()
