#!/usr/bin/env python3

from ckine.fit import build_model
import pymc3 as pm

if __name__ == "__main__": #only go into this loop if you're running fit.py directly instead of running a file that calls fit.py
    M = build_model()
    M.sampling()
    pm.backends.text.dump("IL2_model_results", M.trace) #instead of pickling data we dump it into file that can be accessed by read_fit_data.py
