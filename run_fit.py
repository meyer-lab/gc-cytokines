#!/usr/bin/env python3

# Fix an error specific to MacOS
if __name__ == "__main__":
    from sys import platform
    from multiprocessing import set_start_method

    if platform == "darwin":
        set_start_method('forkserver')

# Set matplotlib backend so python remains in the background
import matplotlib
matplotlib.use("Agg")
from ckine.fit import build_model
import pymc3 as pm

if __name__ == "__main__": #only go into this loop if you're running fit.py directly instead of running a file that calls fit.py
    M = build_model()
    M.build()
    M.sampling()
    pm.backends.text.dump("IL2_model_results", M.trace) #instead of pickling data we dump it into file that can be accessed by read_fit_data.py
