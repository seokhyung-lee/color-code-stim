import argparse
import os
import warnings
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

from src.color_code_stim import ColorCode

warnings.filterwarnings(
    "ignore", message="A worker stopped while some jobs were given to the executor."
)


def count_fails_above_threshold(
    logical_gaps: np.ndarray, fails: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the number of failures and accepted samples for different gap thresholds.

    Given logical_gaps (1D float array) and fails (1D bool array of the same length),
    it computes two arrays: `num_fails_by_c` and `num_accepted_by_c`.
    For each threshold `c_i` in a predefined range `c`:
    - `num_fails_by_c[i]` stores the count of indices `j` such that
      `logical_gaps[j] >= c_i` and `fails[j]` is True.
    - `num_accepted_by_c[i]` stores the count of indices `j` such that
      `logical_gaps[j] >= c_i`.

    Parameters
    ----------
    logical_gaps : np.ndarray
        1D array of float values representing logical gaps.
    fails : np.ndarray
        1D array of boolean values indicating failures, same length as logical_gaps.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - c: The array of thresholds used for comparison.
        - num_fails_by_c: The array where each element i contains the count
          of fails with logical_gaps >= c[i].
        - num_accepted_by_c: The array where each element i contains the count
          of samples with logical_gaps >= c[i].
    """
    # Define the thresholds c
    # Ensure c includes 0 and goes up to the maximum gap, rounded
    max_gap = logical_gaps.max() if logical_gaps.size > 0 else 0
    c = np.arange(0, max_gap + 0.01, 0.01, dtype="float64").round(
        decimals=2
    )  # Add 0.01 to include max_gap if it's a multiple of 0.01

    # Calculate num_accepted_by_c
    if logical_gaps.size == 0:
        num_accepted_by_c = np.zeros_like(c, dtype=int)
    else:
        # Use broadcasting to compare each logical_gap with each threshold in c
        comparison_matrix_all = logical_gaps[:, np.newaxis] >= c
        # Sum the boolean values along axis=0
        num_accepted_by_c = np.sum(comparison_matrix_all, axis=0)

    # Filter the logical gaps where fails is True
    failed_gaps = logical_gaps[fails]

    # If there are no failures, the result is an array of zeros
    if failed_gaps.size == 0:
        num_fails_by_c = np.zeros_like(c, dtype=int)
    else:
        # Use broadcasting to compare each failed_gap with each threshold in c
        # failed_gaps[:, np.newaxis] creates a column vector (M, 1) where M = failed_gaps.size
        # c is a row vector (1, N) implicitly, where N = c.size
        # comparison_matrix[j, i] is True if failed_gaps[j] >= c[i]
        comparison_matrix_fails = failed_gaps[:, np.newaxis] >= c

        # Sum the boolean values along the axis corresponding to failed_gaps (axis=0)
        # This counts how many failed_gaps meet the condition for each c[i]
        num_fails_by_c = np.sum(comparison_matrix_fails, axis=0)

    return c, num_fails_by_c, num_accepted_by_c


def task(shots_batch, dcult, dm, p):
    assert dm > dcult
    cc = ColorCode(
        circuit_type="cult+growing",
        d=dcult,
        d2=dm,
        rounds=dm,
        p_circuit=p,
        comparative_decoding=True,
        perfect_init_final=True,
    )
    det, obs = cc.sample(shots_batch)
    obs_preds, extra_outputs = cc.decode(det, full_output=True)
    cult_success = extra_outputs["cult_success"]
    logical_gaps = extra_outputs["logical_gaps"]
    fails = obs[cult_success] ^ obs_preds

    num_cult_succ = fails.shape[0]
    clist, num_fails_by_c, num_accepted_by_c = count_fails_above_threshold(
        logical_gaps, fails
    )
    df_results = pd.DataFrame(
        {"c": clist, "num_fails": num_fails_by_c, "num_accepted": num_accepted_by_c}
    )
    df_results.set_index("c", inplace=True)

    return num_cult_succ, df_results


def task_parallel(
    shots_to_run, dcult, dm, p, n_jobs=-1, repeat=10
) -> Tuple[int, pd.DataFrame]:
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
        delayed(task)(shots, dcult, dm, p) for shots in job_shots
    )

    # Merge the results
    num_cult_succ = sum(result[0] for result in results)

    # Concatenate all result DataFrames (index='c', columns=['num_fails', 'num_accepted'])
    all_results_df = pd.concat([result[1] for result in results])

    # Group by index 'c' and sum the columns 'num_fails' and 'num_accepted'
    merged_results_df = all_results_df.groupby(all_results_df.index).sum()

    return num_cult_succ, merged_results_df


def run_simulation(shots, p, dcult, dm, n_jobs=-1, repeat=100):
    current_index = (p, dcult, dm)

    # Define filenames
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    cult_succ_filename = os.path.join(data_dir, "df_cult_succ.csv")
    logical_errors_filename = os.path.join(
        data_dir, f"df_logical_errors_p{p}_dcult{dcult}_dm{dm}.csv"
    )

    # Load or initialize df_cult_succ
    try:
        df_cult_succ = pd.read_csv(cult_succ_filename, index_col=[0, 1, 2])
        # Ensure index names are set correctly after loading
        df_cult_succ.index.names = ["p", "dcult", "dm"]
    except FileNotFoundError:
        index = pd.MultiIndex(
            levels=[[], [], []], codes=[[], [], []], names=["p", "dcult", "dm"]
        )
        df_cult_succ = pd.DataFrame(columns=["shots", "num_cult_succ"], index=index)
        # Ensure column types are appropriate if df is empty
        df_cult_succ = df_cult_succ.astype({"shots": int, "num_cult_succ": int})

    # Load or initialize df_logical_errors for the current (p, dcult, dm)
    try:
        df_logical_errors = pd.read_csv(logical_errors_filename, index_col="c")
    except FileNotFoundError:
        df_logical_errors = pd.DataFrame(
            columns=["num_fails", "num_accepted"],
            index=pd.Index([], name="c"),
        )
        # Ensure column type is appropriate if df is empty
        df_logical_errors = df_logical_errors.astype(
            {"num_fails": int, "num_accepted": int}
        )

    # Check shots already done
    shots_done = 0
    if current_index in df_cult_succ.index:
        # Ensure accessing scalar value correctly
        shots_done = int(df_cult_succ.loc[current_index, "shots"])

    # Calculate remaining shots
    shots_to_run_now = (
        shots - shots_done
    )  # Use original 'shots' variable for total required

    if shots_to_run_now > 0:
        # Print current timestamp before starting the simulation
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"\n[{current_time}] Running simulation for (p={p}, dcult={dcult}, dm={dm}) with {shots_to_run_now} shots..."
        )
        # Ensure shots_to_run_now is integer for the simulation function
        num_cult_succ_new, results_df_new = task_parallel(
            int(shots_to_run_now),
            dcult,
            dm,
            p,
            n_jobs=n_jobs,
            repeat=repeat,
        )

        # Update df_cult_succ
        if current_index in df_cult_succ.index:
            df_cult_succ.loc[current_index, "shots"] = int(
                df_cult_succ.loc[current_index, "shots"]
            ) + int(shots_to_run_now)
            df_cult_succ.loc[current_index, "num_cult_succ"] = int(
                df_cult_succ.loc[current_index, "num_cult_succ"]
            ) + int(num_cult_succ_new)
        else:
            new_row = pd.DataFrame(
                {
                    "shots": [int(shots_to_run_now)],
                    "num_cult_succ": [int(num_cult_succ_new)],
                },
                index=pd.MultiIndex.from_tuples(
                    [current_index], names=["p", "dcult", "dm"]
                ),
            )
            df_cult_succ = pd.concat([df_cult_succ, new_row])
            # Ensure correct types after concat if initial df was empty
            df_cult_succ = df_cult_succ.astype({"shots": int, "num_cult_succ": int})

        # Update df_logical_errors
        # Ensure index types are consistent (numeric) before combining
        df_logical_errors.index = pd.to_numeric(df_logical_errors.index)
        results_df_new.index = pd.to_numeric(results_df_new.index)

        # Use pandas `add` method for element-wise addition, aligning indices and filling missing values with 0
        df_logical_errors = df_logical_errors.add(results_df_new, fill_value=0)

        # Ensure the result column types are integer
        df_logical_errors["num_fails"] = df_logical_errors["num_fails"].astype(int)
        df_logical_errors["num_accepted"] = df_logical_errors["num_accepted"].astype(
            int
        )

        # Save updated dataframes
        df_cult_succ.sort_index(inplace=True)
        df_cult_succ.to_csv(cult_succ_filename)

        df_logical_errors.sort_index(inplace=True)
        df_logical_errors.to_csv(logical_errors_filename)
        print(
            f"(p={p}, dcult={dcult}, dm={dm}): Ran {int(shots_to_run_now)} shots. "
            f"Total shots: {int(df_cult_succ.loc[current_index, 'shots'])}. "
            f"New cult succ: {int(num_cult_succ_new)}. "
            f"Total cult succ: {int(df_cult_succ.loc[current_index, 'num_cult_succ'])}."
        )

    else:
        print(
            f"(p={p}, dcult={dcult}, dm={dm}): Already completed {shots_done}/{shots} shots. Skipping."
        )


if __name__ == "__main__":
    for shots in np.arange(round(1e7), round(1e8) + 1, round(1e7)):
        for p in [1e-3, 5e-4]:
            for dcult in [3, 5]:
                dm_max = 19 if p == 1e-3 else 17
                for dm in range(dcult + 2, dm_max + 1, 4):
                    run_simulation(shots, p, dcult, dm, n_jobs=19, repeat=100)

    print("\nAll simulations finished.")
