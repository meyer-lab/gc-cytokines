#!/usr/bin/env python3

import argparse
from ckine.fit import sampling
import pymc3 as pm

if __name__ == "__main__":  # only go into this loop if you're running fit.py directly instead of running a file that calls fit.py

    parser = argparse.ArgumentParser(
        description='Fit model to experimental data to obtain rate parameters',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--cytokines', required=True, dest='cytokines',
                        help='2 if you want IL2/15 model, 4 if you want IL4/7 model.')
    parser.add_argument('-t', '--trafficking', required=True, dest='trafficking',
                        help='T if we want to use the trafficking model, F if we want no traf model.')

    args = parser.parse_args()
    if args.trafficking == "T":
        traf = True
    elif args.trafficking == "F":
        traf = False
    else:
        raise ValueError("Trafficking should be true or false.")

    if args.cytokines == '2':
        from ckine.fit import build_model
        if traf:
            print('Running IL2/15 model with trafficking')
            filename = "ckine/data/fits/IL2_model_results"
        else:
            print('Running IL2/15 model without trafficking')
            filename = "ckine/data/fits/IL2_15_no_traf"
        M = build_model(traf=traf)
    elif args.cytokines == '4':
        from ckine.fit_others import build_model
        if traf:
            print('Running IL4/7 model with trafficking')
            filename = "ckine/data/fits/IL4-7_model_results"
            M = build_model()
        else:
            raise ValueError("We're only looking at the IL-4/7 fit with trafficking.")
    else:
        raise ValueError("Enter valid fit selection.")

    trace = sampling(M.M)
    pm.backends.text.dump(filename, trace)  # instead of pickling data we dump it into file that can be accessed by read_fit_data.py
