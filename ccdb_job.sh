#!/bin/bash
#SBATCH --account=def-jacobsen  # The group's allocation
#SBATCH -N 1                    # Number of nodes (computers) (usually 1)
#SBATCH -n 32                    # Number of CPU cores (processors)
#SBATCH --mem-per-cpu=512M      # Memory per CPU core (e.g., 512M, 1G, 4G)
                                # (Don't use this specific option on Niagara)
#SBATCH --time=00-00:20:00      # Max runtime (Days-HH:MM:SS) - JOB KILLED AFTER THIS!
#SBATCH --job-name=SmallQFT # Descriptive job name
#SBATCH --output=/project/def-jacobsen/zachvern/logs/%x-%j.out  # Standard output file (%x=jobname, %j=jobid)
                                                      # Ensure '/path/to/your/project/logs' exists on HPC!
#SBATCH --error=/project/def-jacobsen/zachvern/logs/%x-%j.err   # Standard error file
                                                      # Ensure '/path/to/your/project/logs' exists on HPC!
#SBATCH --mail-user=zachary.vernec@mail.utoronto.ca # Optional: Email for job status
#SBATCH --mail-type=FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT #,END # Email events

echo "Job started on $(hostname) at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "My working directory on HPC is: $(pwd)" # This will be your submission directory


echo "--- Setting up software ---"
module load StdEnv/2020
module load python/3.11
module load kahypar
module load scipy-stack/2025a

# Create a temporary, private workspace for this job on the HPC
# $SLURM_TMPDIR is a special fast, temporary directory for your job on the HPC
# Using it is often faster for installing packages and running code
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate" # Activate virtual environment

echo "--- Installing Python packages ---"
# Using --no-index tells pip not to look online, good if pre-downloaded wheels are available
# See: https://docs.alliancecan.ca/wiki/Python#Installing_packages
pip install --no-index --upgrade pip
pip install --no-index -r "/project/def-jacobsen/zachvern/requirements.txt"

echo "--- Running my Python script ---"
# Replace with your actual command. Ensure paths to script/data are correct on HPC.
python "/project/def-jacobsen/zachvern/main.py" #--input /path/to/data --output "$SLURM_TMPDIR/results.csv"

SCRIPT_EXIT_CODE=$? # Save script's exit code
echo "--- Python script finished with exit code $SCRIPT_EXIT_CODE ---"

# If your script created important files in $SLURM_TMPDIR, copy them back to persistent storage
echo "--- Copying data to home (/project/def-jacobsen/zachvern/) ---"
cp -v "$SLURM_TMPDIR/results-data-{gcp-s,gcp-e,fgp-roee,mlfm-r,zv-thy,pytket-de,pytket-aesd}.txt" "/project/def-jacobsen/zachvern"

echo "--- Job finished at: $(date) ---"
exit $SCRIPT_EXIT_CODE # Important: Exit with your script's exit code