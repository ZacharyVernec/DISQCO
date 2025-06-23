#!/bin/bash
#SBATCH --account=def-jacobsen  # The group's allocation
#SBATCH -N 1                    # Number of nodes (computers) (usually 1)
#SBATCH -n 16                    # Number of CPU cores (processors)
#SBATCH --mem-per-cpu=512M      # Memory per CPU core (e.g., 512M, 1G, 4G)
                                # (Don't use this specific option on Niagara)
#SBATCH --time=00-00:01:30      # Max runtime (Days-HH:MM:SS) - JOB KILLED AFTER THIS!
#SBATCH --job-name=SmallQFT # Descriptive job name
#SBATCH --output=/home/zachvern/logs/%x-%j.out  # Standard output file (%x=jobname, %j=jobid)
                                                      # Ensure '/path/to/your/project/logs' exists on HPC!
#SBATCH --error=/home/zachvern/logs/%x-%j.err   # Standard error file
                                                      # Ensure '/path/to/your/project/logs' exists on HPC!
#SBATCH --mail-user=zachary.vernec@mail.utoronto.ca # Optional: Email for job status
#SBATCH --mail-type=FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT #,END # Email events

echo "Job started on $(hostname) at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "My working directory on HPC is: $(pwd)" # This will be your submission directory

echo "--- Setting up software ---"
module load apptainer

apptainer run -C -W ${SLURM_TMPDIR} image.sif

# If your script created important files in $SLURM_TMPDIR, copy them back to persistent storage
echo "--- Copying data to home (/home/zachvern/) ---"
echo "Data in SLURM_TMPDIR: "
ls -l "$SLURM_TMPDIR"
echo "Data in SLURM_TMPDIR (recursive): "
ls -lR
echo "Copy: "
cp -v "$SLURM_TMPDIR/tmp/results-data-{gcp-s,gcp-e,fgp-roee,mlfm-r,zv-thy,pytket-de,pytket-aesd}.txt" "/home/zachvern/"

echo "--- Job finished at: $(date) ---"
exit $SCRIPT_EXIT_CODE # Important: Exit with your script's exit code