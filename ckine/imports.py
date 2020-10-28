"""File that deals with everything about importing and sampling."""
from glob import iglob
from os.path import join, dirname
import numpy as np
from scipy.stats import gmean
import pandas as pds
from .model import nParams

path_here = dirname(dirname(__file__))


def import_Rexpr():
    """ Loads CSV file containing Rexpr levels from Visterra data. """
    data = pds.read_csv(join(path_here, "ckine/data/final_receptor_levels.csv"))  # Every row in the data represents a specific cell

    with np.errstate(divide="ignore"):
        df = data.groupby(["Cell Type", "Receptor"]).agg(gmean)  # Get the mean receptor count for each cell across trials in a new dataframe.

    cell_names, receptor_names = df.index.unique().levels  # gc_idx=0|IL15Ra_idx=1|IL2Ra_idx=2|IL2Rb_idx=3
    cell_names = cell_names[[4, 0, 5, 1, 9, 7, 3, 8, 6, 2]]  # Reorder to match pstat import order
    receptor_names = receptor_names[[2, 3, 0, 1, 4]]  # Reorder so that IL2Ra_idx=0|IL2Rb_idx=1|gc_idx=2|IL15Ra_idx=3|IL7Ra_idx=4
    numpy_data = pds.Series(df["Count"]).values.reshape(cell_names.size, receptor_names.size)  # Rows are in the order of cell_names. Receptor Type is on the order of receptor_names
    numpy_data = numpy_data[:, [2, 3, 0, 1, 4]]  # Rearrange numpy_data to place IL2Ra first, then IL2Rb, then gc, then IL15Ra in this order
    numpy_data = numpy_data[[4, 0, 5, 1, 9, 7, 3, 8, 6, 2], :]  # Reorder to match cells
    return data, numpy_data, cell_names


def import_muteins():
    """ Import mutein data and return a normalized DataFrame and tensor. """
    data = pds.read_csv(join(path_here, "ckine/data/2019-07-mutein-timecourse.csv"))

    # Concentrations are across columns, so melt
    data = pds.melt(data, id_vars=["Cells", "Ligand", "Time", "Replicate"], var_name="Concentration", value_name="RFU")

    # Make the concentrations numeric
    data["Concentration"] = pds.to_numeric(data["Concentration"])

    # Subtract off the minimum signal
    data["RFU"] = data["RFU"] - data.groupby(["Cells", "Replicate"])["RFU"].transform("min")

    # Each replicate varies in its sensitivity, so correct for that
    replAvg = data[data["Time"] > 0.6].groupby(["Replicate"]).mean()
    ratio = replAvg.loc[2, "RFU"] / replAvg.loc[1, "RFU"]
    data.loc[data["Replicate"] == 1, "RFU"] *= ratio

    # Take the average across replicates
    dataMean = data.groupby(["Cells", "Ligand", "Time", "Concentration"]).mean()
    dataMean.drop("Replicate", axis=1, inplace=True)

    # Make a data tensor. Dimensions correspond to groupby above
    dataTensor = np.reshape(dataMean["RFU"].values, (9, 4, 4, 12))

    return dataMean, dataTensor


def loadFiles(pathh):
    """ Load files as a dataframe. """
    all_files = iglob(join(path_here, "ckine/data/fits/", pathh, "*.csv"))

    return pds.concat((pds.read_csv(f) for f in all_files))


def import_samples_2_15(Traf=True, ret_trace=False, N=None, tensor=False):
    """ This function imports the csv results of IL2-15 fitting into a numpy array called unkVec. """
    if tensor:
        np.random.seed(1)

    if Traf:
        trace = loadFiles("IL2_model_results")
    else:
        trace = loadFiles("IL2_15_no_traf")

    # option to return trace instead of numpy array
    if ret_trace:
        return trace

    num = trace["kfwd__0"].values.size

    unkVec = np.zeros((nParams(), num))
    unkVec[6:13, :] = trace[["kfwd__0", "rxn__0", "rxn__1", "rxn__2", "rxn__3", "rxn__4", "rxn__5"]].values.T
    unkVec[13:17, :] = 1.0
    unkVec[22:26, :] = trace[["Rexpr_2Ra__0", "Rexpr_2Rb__0", "Rexpr_gc__0", "Rexpr_15Ra__0"]].values.T

    if Traf:
        unkVec[17:22, :] = trace[["endo__0", "activeEndo__0", "sortF__0", "kRec__0", "kDeg__0"]].T

    if N is not None:
        assert 0 < N < num, "The N specified is out of bounds."

        idx = np.random.randint(num, size=N)  # pick N numbers without replacement from 0 to num
        unkVec = unkVec[:, idx]

    return unkVec


def import_samples_4_7(ret_trace=False, N=None):
    """ This function imports the csv results of IL4-7 fitting into a numpy array called unkVec. """
    trace = loadFiles("IL4-7_model_results")

    # option to return trace instead of numpy array
    if ret_trace:
        return trace

    endo = trace["endo__0"].values
    sortF = trace["sortF__0"].values
    kRec = trace["kRec__0"].values
    kDeg = trace["kDeg__0"].values
    num = endo.shape[0]

    unkVec = np.zeros((nParams(), num))
    unkVec[6, :] = trace["kfwd__0"].values
    unkVec[7:17, :] = 1.0
    unkVec[13, :] = trace["k27rev__0"].values
    unkVec[15, :] = trace["k33rev__0"].values
    unkVec[17, :] = endo
    unkVec[18, :] = trace["activeEndo__0"].values
    unkVec[19, :] = sortF
    unkVec[20, :] = kRec
    unkVec[21, :] = kDeg

    # Constant according to measured number per cell
    unkVec[24, :] = (328.0 * endo) / (1.0 + ((kRec * (1.0 - sortF)) / (kDeg * sortF)))  # gc
    unkVec[26, :] = (2591.0 * endo) / (1.0 + ((kRec * (1.0 - sortF)) / (kDeg * sortF)))  # IL-7R
    unkVec[28, :] = (254.0 * endo) / (1.0 + ((kRec * (1.0 - sortF)) / (kDeg * sortF)))  # IL-4R

    if N is not None:
        assert 0 < N < num, "The N specified is out of bounds."

        idx = np.random.randint(num, size=N)  # pick N numbers without replacement from 0 to num
        unkVec = unkVec[:, idx]

    return unkVec


def import_pstat(combine_samples=True):
    """ Loads CSV file containing pSTAT5 levels from Visterra data. Incorporates only Replicate 1 since data missing in Replicate 2. """
    data = np.array(pds.read_csv(join(path_here, "ckine/data/pSTAT_data.csv"), encoding="latin1"))
    ckineConc = data[4, 2:14]
    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    # 4 time points, 10 cell types, 12 concentrations, 2 replicates
    IL2_data = np.zeros((40, 12))
    IL2_data2 = IL2_data.copy()
    IL15_data = IL2_data.copy()
    IL15_data2 = IL2_data.copy()
    cell_names = list()
    for i in range(10):
        cell_names.append(data[12 * i + 3, 1])
        # Subtract the zero treatment plates before assigning to returned arrays
        if i <= 4:
            zero_treatment = data[12 * (i + 1), 13]
            zero_treatment2 = data[8 + (12 * i), 30]
        else:
            zero_treatment = data[8 + (12 * i), 13]
            zero_treatment2 = data[8 + (12 * i), 30]
        # order of increasing time by cell type
        IL2_data[4 * i: 4 * (i + 1), :] = np.flip(data[6 + (12 * i): 10 + (12 * i), 2:14].astype(np.float) - zero_treatment, 0)
        IL2_data2[4 * i: 4 * (i + 1), :] = np.flip(data[6 + (12 * i): 10 + (12 * i), 19:31].astype(np.float) - zero_treatment2, 0)
        IL15_data[4 * i: 4 * (i + 1), :] = np.flip(data[10 + (12 * i): 14 + (12 * i), 2:14].astype(np.float) - zero_treatment, 0)
        IL15_data2[4 * i: 4 * (i + 1), :] = np.flip(data[10 + (12 * i): 14 + (12 * i), 19:31].astype(np.float) - zero_treatment2, 0)

    if combine_samples is False:
        return ckineConc, cell_names, IL2_data, IL2_data2, IL15_data, IL15_data2

    for i in range(IL2_data.shape[0]):
        for j in range(IL2_data.shape[1]):
            # take average of both replicates if specific entry isn't nan
            IL2_data[i, j] = np.nanmean(np.array([IL2_data[i, j], IL2_data2[i, j]]))
            IL15_data[i, j] = np.nanmean(np.array([IL15_data[i, j], IL15_data2[i, j]]))

    dataMean = pds.DataFrame(
        {
            "Cells": np.tile(np.repeat(cell_names, 48), 2),
            "Ligand": np.concatenate((np.tile(np.array("IL2"), 480), np.tile(np.array("IL15"), 480))),
            "Time": np.tile(np.repeat(tps, 12), 20),
            "Concentration": np.tile(ckineConc, 80),
            "RFU": np.concatenate((IL2_data.reshape(480), IL15_data.reshape(480))),
        }
    )

    for conc in dataMean.Concentration.unique():
        dataMean = dataMean.replace({"Concentration": conc}, np.round(conc, decimals=9))

    return ckineConc, cell_names, IL2_data, IL15_data, dataMean
