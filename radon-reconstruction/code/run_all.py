"""Reproduce every table, figure and number of the report.

    python code/run_all.py            # sweep, spectral analysis, supplementary runs, figures
    python code/run_all.py figures    # redraw the figures from results/

Data are simulated on a 4x finer grid (400 x 400 pixels, 4x finer detector) and
binned to the 100 x 100 reconstruction model, so the reconstruction operator is
not the one that generated the data.  All noise realisations use fixed seeds."""

import sys
import time

import ablations
import figures
import spectral
import sweep

if __name__ == "__main__":
    t0 = time.time()
    if len(sys.argv) == 1:
        sweep.main()
        spectral.main()
        ablations.main()
    figures.main()
    print(f"total {time.time() - t0:.0f} s")
