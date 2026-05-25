#!/bin/bash

# Define parameter ranges
seeds=(0 10 20 30 40 50 60 70 80 90)
n_tests=(100)
n_trains=(50)
Ms=(2)
num_steps=500
repetitions=10
validity="True"

validity_args=()
if [[ "$validity" == "True" ]]; then
  validity_args=(--validity)
fi

for M in "${Ms[@]}"; do
  for n_test in "${n_tests[@]}"; do
    for n_train in "${n_trains[@]}"; do
      for seed in "${seeds[@]}"; do
        job_name="multi_hc_M${M}_nt${n_test}_ntr${n_train}_s${seed}"

        # Submit the job
        sbatch --job-name="$job_name" \
               submit_multivariate_sbatch.sh \
               --M "$M" \
               --n_test "$n_test" \
               --n_train "$n_train" \
               --seed "$seed" \
               --steps "$num_steps" \
               --repetitions "$repetitions" \
               "${validity_args[@]}"

        echo "Submitted job: $job_name"
      done
    done
  done
done
