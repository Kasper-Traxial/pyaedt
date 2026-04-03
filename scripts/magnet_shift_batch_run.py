# Copyright (C) 2021 - 2026 ANSYS, Inc. and/or its affiliates.
# SPDX-License-Identifier: MIT
"""Batch runner for magnet-shift optimisation results.

For every combination of:
  - segment count  : 2, 3, 4, 5
  - variant type   : magnet_shift | WidthSharing_magnet_shift |
                     Contained_magnet_shift | Contained_and_WidthSharing_magnet_shift

the script reads the three optimal-parameter rows (Min Combined / Min THD /
Min Ripple) from the matching ``<N>_segments_<variant>_optimal_parameters.csv``
file, sets the corresponding design variables (s1..sN, w1..wN) in the AEDT
project, re-runs Setup1, and exports the ``Torque Plot1`` and ``Backemf``
reports to CSV.

Results from the three optimisation variants belonging to the *same* source CSV
are merged into a single output file per report, e.g.

    2_segments_Contained_magnet_shift_Torque_Plot1_results.csv
    2_segments_Contained_magnet_shift_Backemf_results.csv

Prerequisites
-------------
* Ansys Electronics Desktop must already be running and the project
  ``D:\\EM\\132frame - 7.5kw\\<project>.aedt`` must be open, containing
  designs named ``2segments``, ``3segments``, ``4segments``, ``5segments``.
* ``ansys-aedt-core`` (pyaedt) must be installed in the Python environment.
* ``pandas`` must be installed in the Python environment.

Usage
-----
    python magnet_shift_batch_run.py

Configuration
-------------
Edit the constants in the ``# --- Configuration ---`` block below to match
your environment (paths, AEDT version, etc.).
"""

import os
import shutil
import tempfile

import pandas as pd

from ansys.aedt.core import Maxwell3d

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder that contains the optimal-parameter CSV files.
SOURCE_DIR = r"S:\1. NPD\T32-0120-M11-7.5kW Air Cooled\EM\EMD132.6 - 7.5kW"

# Folder that contains the .aedt project file.
PROJECT_DIR = r"D:\EM\132frame - 7.5kw"

# Folder where the output result CSV files will be written.
# Set to SOURCE_DIR to save them alongside the input files.
OUTPUT_DIR = SOURCE_DIR

# AEDT version string, e.g. "2024.2", "2025.1", "2026.1".
AEDT_VERSION = "2026.1"

# ---------------------------------------------------------------------------
# Constants – these should not normally need to be changed
# ---------------------------------------------------------------------------

SEGMENT_COUNTS = [2, 3, 4, 5]

VARIANTS = [
    "magnet_shift",
    "WidthSharing_magnet_shift",
    "Contained_magnet_shift",
    "Contained_and_WidthSharing_magnet_shift",
]

# Row labels inside the optimal-parameters CSV files.
OPTIMISATION_ROWS = ["Min Combined", "Min THD", "Min Ripple"]

# Report names that exist in every design.
REPORTS = ["Torque Plot1", "Backemf"]

# Map from segment count → AEDT design name.
DESIGN_MAP = {
    2: "2segments",
    3: "3segments",
    4: "4segments",
    5: "5segments",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_aedt_file(directory: str) -> str:
    """Return the path to the first ``.aedt`` file found in *directory*.

    Raises
    ------
    FileNotFoundError
        If no ``.aedt`` file exists in the directory.
    """
    for entry in os.listdir(directory):
        if entry.lower().endswith(".aedt"):
            return os.path.join(directory, entry)
    raise FileNotFoundError(f"No .aedt project file found in: {directory!r}")


def _source_csv_path(n_segments: int, variant: str) -> str:
    """Return the full path of the optimal-parameters CSV for *n_segments* / *variant*."""
    filename = f"{n_segments}_segments_{variant}_optimal_parameters.csv"
    return os.path.join(SOURCE_DIR, filename)


def _set_design_variables(m3d: Maxwell3d, row: pd.Series, n_segments: int) -> None:
    """Set ``s1``..``sN`` and ``w1``..``wN`` from *row*.

    Parameters
    ----------
    m3d:
        Active Maxwell3d application object.
    row:
        A pandas Series whose index includes ``sliceN_shift`` and
        ``sliceN_width`` for N in ``1..n_segments``.
    n_segments:
        Number of slices/segments for the current design.
    """
    for i in range(1, n_segments + 1):
        shift_col = f"slice{i}_shift"
        width_col = f"slice{i}_width"
        m3d[f"s{i}"] = str(float(row[shift_col]))
        m3d[f"w{i}"] = str(float(row[width_col]))


def _export_report(m3d: Maxwell3d, report_name: str, export_dir: str) -> pd.DataFrame | None:
    """Export *report_name* to a temporary CSV and return it as a DataFrame.

    Parameters
    ----------
    m3d:
        Active Maxwell3d application object.
    report_name:
        Name of the report in AEDT (e.g. ``"Torque Plot1"``).
    export_dir:
        Directory where the temporary CSV file will be written.

    Returns
    -------
    pd.DataFrame | None
        The report data as a DataFrame, or ``None`` if the export fails.
    """
    exported_path = m3d.post.export_report_to_file(
        output_dir=export_dir,
        plot_name=report_name,
        extension=".csv",
        unique_file=True,
    )
    if not exported_path or not os.path.isfile(exported_path):
        print(f"    [WARNING] Export of report '{report_name}' produced no file.")
        return None

    try:
        df = pd.read_csv(exported_path)
        return df
    finally:
        try:
            os.remove(exported_path)
        except OSError:
            pass


def _safe_report_name(report_name: str) -> str:
    """Convert a report name to a filesystem-safe string (replace spaces)."""
    return report_name.replace(" ", "_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full batch simulation loop (48 runs total)."""
    # Locate the AEDT project file.
    aedt_file = _find_aedt_file(PROJECT_DIR)
    project_name = os.path.splitext(os.path.basename(aedt_file))[0]
    print(f"Using project: {aedt_file!r}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Temporary directory for raw report exports (cleaned up at the end).
    temp_dir = tempfile.mkdtemp(prefix="aedt_export_")

    # Connect to AEDT ONCE before the loop.  Creating Maxwell3d multiple times
    # inside the loop would try to re-initialise a gRPC Desktop session for
    # each iteration, which deadlocks when the script is executed from within
    # AEDT ("Run PyAEDT Script") because AEDT is single-threaded and already
    # busy running the script.
    m3d = Maxwell3d(
        project=project_name,
        design=DESIGN_MAP[SEGMENT_COUNTS[0]],
        version=AEDT_VERSION,
        new_desktop=False,
        close_on_exit=False,
    )

    try:
        for n_segments in SEGMENT_COUNTS:
            design_name = DESIGN_MAP[n_segments]

            # Switch the active design (reuses the existing gRPC connection).
            print(f"\nSwitching to design: {design_name!r}")
            m3d.set_active_design(design_name)

            for variant in VARIANTS:
                csv_path = _source_csv_path(n_segments, variant)
                csv_basename = os.path.basename(csv_path)

                if not os.path.isfile(csv_path):
                    print(f"[SKIP] Source CSV not found: {csv_path!r}")
                    continue

                print(f"\n{'='*70}")
                print(f"Processing: {csv_basename}")
                print(f"  Design  : {design_name}")
                print(f"{'='*70}")

                # Read the optimal-parameters CSV.
                params_df = pd.read_csv(csv_path, index_col=0)

                # Per-report accumulator: list of DataFrames, one per opt. row.
                report_frames: dict[str, list[pd.DataFrame]] = {r: [] for r in REPORTS}

                for opt_row_label in OPTIMISATION_ROWS:
                    if opt_row_label not in params_df.index:
                        print(f"  [SKIP] Row '{opt_row_label}' not found in {csv_basename!r}")
                        continue

                    row = params_df.loc[opt_row_label]
                    print(f"\n  Optimisation: {opt_row_label}")

                    # --- 1. Set design variables ---
                    _set_design_variables(m3d, row, n_segments)
                    print(f"    Variables set for {n_segments} segments.")

                    # --- 2. Analyse Setup1 ---
                    print("    Running Setup1 ...")
                    success = m3d.analyze(setup="Setup1", blocking=True)
                    if not success:
                        print(f"    [WARNING] Analysis failed for '{opt_row_label}' – skipping export.")
                        continue
                    print("    Setup1 complete.")

                    # --- 3. Export reports ---
                    for report_name in REPORTS:
                        df = _export_report(m3d, report_name, temp_dir)
                        if df is not None:
                            # Tag the data with the optimisation variant.
                            df.insert(0, "Optimisation", opt_row_label)
                            report_frames[report_name].append(df)
                            print(f"    Exported '{report_name}' ({len(df)} rows).")

                # --- 4. Write combined output CSVs ---
                base_name = os.path.splitext(csv_basename)[0]  # strip .csv
                for report_name, frames in report_frames.items():
                    if not frames:
                        print(f"  [WARNING] No data to save for report '{report_name}'.")
                        continue
                    combined = pd.concat(frames, ignore_index=True)
                    out_filename = f"{base_name}_{_safe_report_name(report_name)}_results.csv"
                    out_path = os.path.join(OUTPUT_DIR, out_filename)
                    combined.to_csv(out_path, index=False)
                    print(f"\n  Saved: {out_path!r}")

    finally:
        # Clean up temporary export directory.
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nBatch run complete.")


if __name__ == "__main__":
    main()
