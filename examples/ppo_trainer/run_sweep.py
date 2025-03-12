import subprocess
import itertools
import os


# Define the hyperparameter grid values
tp_sizes   = [8]             # Example tensor model parallel sizes
pp_sizes   = [4]         # Example pipeline model parallel sizes
dp_sizes   = [2]           # Example data parallel sizes
gen_lens   = [256]     # Example generation lengths
bsz_per_devices = [1]          # Example batch sizes per device
rollout_ns = [1,2,4,8,16,32,64,128,256]                # Example rollout numbers

processes = []

dry_run = False

# Launch all jobs concurrently while ensuring TP * PP * DP <= 64
for tp, pp, dp, gen, bsz_per_device, rollout in itertools.product(tp_sizes, pp_sizes, dp_sizes, gen_lens, bsz_per_devices, rollout_ns):
    if tp * pp * dp > 64:
        continue
    bsz_per_node = bsz_per_device * tp
    print(f"Starting job: TP_SIZE={tp}, PP_SIZE={pp}, DP_SIZE={dp}, GEN_LEN={gen}, BSZ_PER_NODE={bsz_per_node}, ROLLOUT_N={rollout}")
    command = [
        "/home/aiscuser/verl/examples/ppo_trainer/run_profile.sh",
        str(tp),
        str(pp),
        str(dp),
        str(gen),
        str(bsz_per_node),
        str(rollout)
    ]
    if dry_run:
        print(" ".join(command))
        continue
    
    # Launch job, ignoring stdout and stderr
    process = subprocess.Popen(command)
    processes.append(process)

if dry_run:
    print("Dry run complete.")
    exit(0)

# Wait for all jobs to complete (optional)
for p in processes:
    p.wait()

print("All jobs have been fired off.")