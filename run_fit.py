#!/usr/bin/env python3

import sys
from ckine.fit import sampling
import pymc3 as pm

if __name__ == "__main__":  # only go into this loop if you're running fit.py directly instead of running a file that calls fit.py
    if sys.argv[1] == '1':
        print('Running IL2/15 model')
        from ckine.fit import build_model
        filename = "IL2_15_no_traf"
    else:
        print('Running IL4/7 model')
        from ckine.fit_others import build_model
        filename = "IL4-7_model_results"

    M = build_model(traf=False)
    trace = sampling(M.M)
    pm.backends.text.dump(filename, trace)  # instead of pickling data we dump it into file that can be accessed by read_fit_data.py
