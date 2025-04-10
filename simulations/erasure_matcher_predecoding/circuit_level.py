import os
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from src.color_code_stim import ColorCode


def task(shots_batch, d, T, p):
    seed = np.random.randint(0, 2**32)
    cc = ColorCode(d=d, rounds=T, shape="tri", p_circuit=p, comparative_decoding=True)
    nfails_org = cc.simulate(shots_batch, seed=seed)
    nfails_em = cc.simulate(shots_batch, erasure_matcher_predecoding=True, seed=seed)
    nfails_empc = cc.simulate(
        shots_batch,
        erasure_matcher_predecoding=True,
        partial_correction_by_predecoding=True,
        seed=seed,
    )
    return nfails_org, nfails_em, nfails_empc


def run_parallel_simulation(shots_to_run, d, T, p, n_jobs=-1, repeat=4):
    # Determine number of shots per job
    shots_to_run = round(shots_to_run)
    n_jobs = n_jobs if n_jobs > 0 else cpu_count()
    shots_per_job = shots_to_run // (n_jobs * repeat)
    remaining_shots = shots_to_run % (n_jobs * repeat)

    # Distribute shots evenly
    job_shots = [shots_per_job] * (n_jobs * repeat)
    for i in range(remaining_shots):
        job_shots[i] += 1

    # Run parallel jobs
    results = Parallel(n_jobs=n_jobs)(
        delayed(task)(shots, d, T, p) for shots in job_shots
    )

    # Sum up the results
    total_nfails_org = sum(result[0] for result in results)
    total_nfails_em = sum(result[1] for result in results)
    total_nfails_empc = sum(result[2] for result in results)

    return total_nfails_org, total_nfails_em, total_nfails_empc


if __name__ == "__main__":
    total_shots = round(1e7)
    shots_batch = round(1e6)
    p = 0.001
    n_jobs = 18
    repeat = 100
    d_min = 3
    d_max = 19

    for shots in np.arange(0, total_shots + 1, shots_batch)[1:]:
        # Get the current file's directory and create path for results file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_dir, "circuit_level_results.csv")

        # Initialize full_df with existing data if file exists
        if os.path.isfile(filename):
            full_df = pd.read_csv(filename)

        else:
            full_df = pd.DataFrame(
                columns=[
                    "d",
                    "T",
                    "p",
                    "shots",
                    "nfails_org",
                    "nfails_em",
                    "nfails_empc",
                ]
            )

        # Convert column data types to integers and ensure index types are correct
        for col in ["shots", "nfails_org", "nfails_em", "nfails_empc", "d", "T"]:
            full_df[col] = full_df[col].astype(int)
        full_df["p"] = full_df["p"].astype(float)

        if "level_0" in full_df.columns:
            del full_df["level_0"]
        if "index" in full_df.columns:
            del full_df["index"]

        full_df = full_df.set_index(["d", "T", "p"])

        for d in range(d_min, d_max + 1, 2):
            T = 4 * d
            idx = (d, T, p)
            shots_to_run = shots

            print()
            current_time = datetime.now()
            print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

            if idx in full_df.index:
                # Calculate remaining shots to run
                existing_shots = full_df.loc[idx, "shots"]
                shots_to_run = shots - existing_shots

                # Skip if we've already run enough shots
                if shots_to_run <= 0:
                    print(
                        f"Skipping d={d}, T={T}, p={p}, already have {existing_shots} shots (target: {total_shots})"
                    )
                    continue

                print(
                    f"Running {shots_to_run} additional shots for d={d}, T={T}, p={p} (existing: {existing_shots})"
                )
            else:
                print(
                    f"Running {shots_to_run} shots for new configuration d={d}, T={T}, p={p}"
                )

            # Run simulation only for the remaining shots
            nfails_org, nfails_em, nfails_empc = run_parallel_simulation(
                shots_to_run, d, T, p, n_jobs=n_jobs, repeat=repeat
            )
            print(
                f"d={d}, T={T}, p={p}, nfails_org={nfails_org}, nfails_em={nfails_em}, nfails_empc={nfails_empc}"
            )

            # Create new result
            new_result = {
                "shots": round(shots_to_run),
                "nfails_org": round(nfails_org),
                "nfails_em": round(nfails_em),
                "nfails_empc": round(nfails_empc),
            }

            # Update results
            if idx in full_df.index:
                # Add to existing values
                full_df.loc[idx, "shots"] += round(shots_to_run)
                full_df.loc[idx, "nfails_org"] += round(nfails_org)
                full_df.loc[idx, "nfails_em"] += round(nfails_em)
                full_df.loc[idx, "nfails_empc"] += round(nfails_empc)
                print(f"Updated existing data for d={d}, T={T}, p={p}")
            else:
                # Add new row
                full_df.loc[idx, :] = new_result
                print(f"Added new data for d={d}, T={T}, p={p}")

            full_df.sort_index(inplace=True)

            # Save after each iteration
            full_df.to_csv(filename)
            print(f"Data saved to {filename}")

        print(f"Final results saved to {filename}")
