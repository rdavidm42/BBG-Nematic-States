#!/bin/bash
pwd; hostname; date

## code
code=${1}
interaction=${2}

rm params.json
rm tight_binding.json

# Iterate over each value of electric field and number density:
for v in $(seq 25 5 200); do
	for n in $(seq 1e11 .4e11 1.5e12); do
		# Create the folder structure
		folder_name="all_jobs_e=${interaction}/job_v${v}n${n}"

		mkdir -p "$folder_name"

		## move the python file to the directory
		cp "${code}" $folder_name
		cp "solve_sc.py" $folder_name
		cp "get_dispersion_one_v.py" $folder_name

       		## move the slurm runner to the directory
        	cp "slurmrun.sh" $folder_name

        	if [ -d "$folder_name" ]; then
                	echo "Running $model in $folder_name"
			(cd "$folder_name" && sbatch "slurmrun.sh" "${code}" $v $n $interaction)
        	else
            		echo "Directory $folder_name not found."

		fi
	done
done
