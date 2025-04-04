import os

import pandas as pd
from joblib import Parallel, cpu_count, delayed

from src.color_code_stim import ColorCode


def task(shots_batch, d, p):
    cc = ColorCode(d=d, rounds=1, shape="tri", p_bitflip=p, comparative_decoding=True)
    nfails1 = cc.simulate(shots_batch)
    nfails2 = cc.simulate(shots_batch, erasure_matcher_predecoding=True)
    return nfails1, nfails2


def run_parallel_simulation(shots_to_run, d, p, n_jobs=-1, repeat=4):
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
    results = Parallel(n_jobs=n_jobs)(delayed(task)(shots, d, p) for shots in job_shots)

    # Sum up the results
    total_nfails1 = sum(result[0] for result in results)
    total_nfails2 = sum(result[1] for result in results)

    return total_nfails1, total_nfails2


if __name__ == "__main__":
    total_shots = round(1e7)
    p = 0.03
    n_jobs = 19
    repeat = 100
    d_max = 31

    filename = "bitflip_results.csv"

    # Initialize full_df with existing data if file exists
    if os.path.isfile(filename):
        full_df = pd.read_csv(filename)

    else:
        full_df = pd.DataFrame(columns=["d", "p", "shots", "nfails_org", "nfails_em"])

    # Convert column data types to integers and ensure index types are correct
    for col in ["shots", "nfails_org", "nfails_em"]:
        full_df[col] = full_df[col].astype(int)

    full_df = full_df.reset_index()
    full_df["d"] = full_df["d"].astype(int)
    full_df["p"] = full_df["p"].astype(float)

    full_df = full_df.set_index(["d", "p"])

    for d in range(3, d_max + 1, 2):
        # Check if this (d, p) combination already exists
        idx = (d, p)
        shots_to_run = total_shots

        if idx in full_df.index:
            # Calculate remaining shots to run
            existing_shots = full_df.loc[idx, "shots"]
            shots_to_run = total_shots - existing_shots

            # Skip if we've already run enough shots
            if shots_to_run <= 0:
                print(
                    f"Skipping d={d}, p={p}, already have {existing_shots} shots (target: {total_shots})"
                )
                continue

            print(
                f"Running {shots_to_run} additional shots for d={d}, p={p} (existing: {existing_shots})"
            )
        else:
            print(f"Running {shots_to_run} shots for new configuration d={d}, p={p}")

        # Run simulation only for the remaining shots
        nfails_org, nfails_em = run_parallel_simulation(
            shots_to_run, d, p, n_jobs=n_jobs, repeat=repeat
        )
        print(f"d={d}, p={p}, nfails_org={nfails_org}, nfails_em={nfails_em}")

        # Create new result
        new_result = {
            "shots": round(shots_to_run),
            "nfails_org": round(nfails_org),
            "nfails_em": round(nfails_em),
        }

        # Update results
        if idx in full_df.index:
            # Add to existing values
            full_df.loc[idx, "shots"] += shots_to_run
            full_df.loc[idx, "nfails_org"] += nfails_org
            full_df.loc[idx, "nfails_em"] += nfails_em
            print(f"Updated existing data for d={d}, p={p}")
        else:
            # Add new row
            full_df.loc[idx, :] = new_result
            print(f"Added new data for d={d}, p={p}")

        full_df.sort_index(inplace=True)

        # Save after each iteration
        full_df.to_csv(filename)
        print(f"Data saved to {filename}")

    print(f"Final results saved to {filename}")
