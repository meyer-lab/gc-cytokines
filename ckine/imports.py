"""File that deals with everything about importing and sampling."""
import os
from os.path import join
import pymc3 as pm
import numpy as np
import pandas as pds
from .fit import build_model as build_model_2_15, find_gc
from .fit_others import build_model as build_model_4_7
from .model import nParams


def import_Rexpr():
    """ Loads CSV file containing Rexpr levels from Visterra data. """
    path = os.path.dirname(os.path.dirname(__file__))
    data = pds.read_csv(join(path, 'ckine/data/final_receptor_levels.csv'))  # Every row in the data represents a specific cell
    df = data.groupby(['Cell Type', 'Receptor']).mean()  # Get the mean receptor count for each cell across trials in a new dataframe.
    cell_names, receptor_names = df.index.unique().levels  # gc_idx=0|IL15Ra_idx=1|IL2Ra_idx=2|IL2Rb_idx=3
    receptor_names = receptor_names[[2, 3, 0, 1]]  # Reorder so that IL2Ra_idx=0|IL2Rb_idx=1|gc_idx=2|IL15Ra_idx=3
    numpy_data = pds.Series(df['Count']).values.reshape(cell_names.size, receptor_names.size)  # Rows are in the order of cell_names. Receptor Type is on the order of receptor_names
    # Rearrange numpy_data to place IL2Ra first, then IL2Rb, then gc, then IL15Ra in this order
    return data, numpy_data[:, [2, 3, 0, 1]], cell_names


def import_samples_2_15(Traf=True, ret_trace=False, N=None):
    """ This function imports the csv results of IL2-15 fitting into a numpy array called unkVec. """
    bmodel = build_model_2_15(traf=Traf)
    n_params = nParams()

    path = os.path.dirname(os.path.abspath(__file__))

    if Traf:
        trace = pm.backends.text.load(join(path, '../IL2_model_results'), bmodel.M)
    else:
        trace = pm.backends.text.load(join(path, '../IL2_15_no_traf'), bmodel.M)

    # option to return trace instead of numpy array
    if ret_trace:
        return trace

    scales = trace.get_values('scales')
    num = scales.size
    kfwd = trace.get_values('kfwd')
    rxn = trace.get_values('rxn')
    Rexpr_2 = trace.get_values('Rexpr_2Ra_2Rb')
    Rexpr_15 = trace.get_values('Rexpr_15Ra')

    if Traf:
        endo = trace.get_values('endo')
        activeEndo = trace.get_values('activeEndo')
        sortF = trace.get_values('sortF')
        kRec = trace.get_values('kRec')
        kDeg = trace.get_values('kDeg')
        Rexpr_gc = find_gc(Traf, endo, kRec, sortF, kDeg)
    else:
        endo = np.zeros((num))
        activeEndo = np.zeros((num))
        sortF = np.zeros((num))
        kRec = np.zeros((num))
        kDeg = np.zeros((num))
        Rexpr_gc = np.ones((num), dtype=float) * find_gc(Traf, endo, kRec, sortF, kDeg)

    unkVec = np.zeros((n_params, num))
    for ii in range(num):
        unkVec[:, ii] = np.array([0., 0., 0., 0., 0., 0., kfwd[ii], rxn[ii, 0], rxn[ii, 1], rxn[ii, 2], rxn[ii, 3], rxn[ii, 4], rxn[ii, 5], 1., 1., 1., 1., endo[ii],
                                  activeEndo[ii], sortF[ii], kRec[ii], kDeg[ii], Rexpr_2[ii, 0], Rexpr_2[ii, 1], Rexpr_gc[ii], Rexpr_15[ii], 0., 0., 0., 0.])

    if N is not None:
        if 0 < N < num:  # return a subsample if the user specified the number of samples
            idx = np.random.randint(num, size=N)  # pick N numbers without replacement from 0 to num
            unkVec, scales = unkVec[:, idx], scales[idx, :]
        else:
            print("The N specified is out of bounds.")
            raise ValueError

    return unkVec, scales


def import_samples_4_7(N=None):
    ''' This function imports the csv results of IL4-7 fitting into a numpy array called unkVec. '''
    bmodel = build_model_4_7()
    n_params = nParams()

    path = os.path.dirname(os.path.abspath(__file__))
    trace = pm.backends.text.load(join(path, '../IL4-7_model_results'), bmodel.M)
    kfwd = trace.get_values('kfwd')
    k27rev = trace.get_values('k27rev')
    k33rev = trace.get_values('k33rev')
    endo = trace.get_values('endo')
    activeEndo = trace.get_values('activeEndo')
    sortF = trace.get_values('sortF')
    kRec = trace.get_values('kRec')
    kDeg = trace.get_values('kDeg')
    scales = trace.get_values('scales')
    GCexpr = (328. * endo) / (1. + ((kRec * (1. - sortF)) / (kDeg * sortF)))  # constant according to measured number per cell
    IL7Raexpr = (2591. * endo[0]) / (1. + ((kRec * (1. - sortF)) / (kDeg * sortF)))  # constant according to measured number per cell
    IL4Raexpr = (254. * endo) / (1. + ((kRec * (1. - sortF)) / (kDeg * sortF)))  # constant according to measured number per cell
    num = scales.shape[0]

    unkVec = np.zeros((n_params, num))
    for ii in range(num):
        unkVec[:, ii] = np.array([0., 0., 0., 0., 0., 0., kfwd[ii], 1., 1., 1., 1., 1., 1., k27rev[ii], 1., k33rev[ii], 1., endo[ii],
                                  activeEndo[ii], sortF[ii], kRec[ii], kDeg[ii], 0., 0., GCexpr[ii], 0., IL7Raexpr[ii], 0., IL4Raexpr[ii], 0.])

    if N is not None:
        if 0 < N < num:  # return a subsample if the user specified the number of samples
            idx = np.random.randint(num, size=N)  # pick N numbers without replacement from 0 to num
            unkVec, scales = unkVec[:, idx], scales[idx, :]
        else:
            print("The N specified is out of bounds.")
            raise ValueError

    return unkVec, scales


def import_pstat():
    """ Loads CSV file containing pSTAT5 levels from Visterra data. Incorporates only Replicate 1 since data missing in Replicate 2. """
    path = os.path.dirname(os.path.dirname(__file__))
    data = np.array(pds.read_csv(join(path, 'ckine/data/Visterra_pSTAT5.csv'), encoding='latin1'))
    ckineConc = data[4, 2:14]
    # 4 time points, 11 cell types, 12 concentrations
    IL2_data = np.zeros((44, 12))
    IL15_data = np.zeros((44, 12))
    cell_names = list()
    for i in range(11):
        cell_names.append(data[12 * i + 3, 1])
        # Subtract the zero treatment plates before assigning to returned arrays
        if i <= 4:
            zero_treatment = data[12 * (i + 1), 13]
        else:
            zero_treatment = data[8 + (12 * i), 13]
        IL2_data[4 * i:4 * (i + 1), :] = data[6 + (12 * i):10 + (12 * i), 2:14] - zero_treatment
        IL15_data[4 * i:4 * (i + 1), :] = data[10 + (12 * i):14 + (12 * i), 2:14] - zero_treatment

    return ckineConc, cell_names, IL2_data, IL15_data
