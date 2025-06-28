# launcher_exp.py

import os
import glob
import baseline_experiment
from utils_exp import generate_base_command, generate_run_commands

# the list of suite_ids you care about
suites = [334, 335, 336, 337]
SEED = 10  # or whatever default you like

def main():
    command_list = []
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "original_data")

    for suite_id in suites:
        # look for folders like "original_data/336_361088", etc.
        pattern = os.path.join(data_dir, f"{suite_id}_*")
        for folder in glob.glob(pattern):
            # folder == ".../336_361088"
            task_id = int(os.path.basename(folder).split("_", 1)[1])

            # disable Numba on-disk cache for each sbatch-wrapped job
            env_prefix = (
                "export NUMBA_DISABLE_JIT=1 && "
                "export NUMBA_CACHE_DIR=/tmp/numba_nocache && "
            )

            # build the base python command with flags
            python_cmd = generate_base_command(
                baseline_experiment,
                flags={
                    'suite_id': suite_id,
                    'task_id': task_id,
                    'result_folder': 'result_folder',
                    'seed': SEED
                }
            )

            # prepend the env exports
            cmd = env_prefix + python_cmd
            command_list.append(cmd)

    # now submit them to Euler via sbatch, with the env vars baked in
    generate_run_commands(
        command_list,
        promt=False,
        num_cpus=1,
        mem=32 * 1024,
        duration="119:59:00",
        mode="euler"
    )

if __name__ == "__main__":
    main()
