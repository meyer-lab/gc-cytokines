"""
This creates Figure 4. Comparison of Experimental verus Predicted Activity across IL2 and IL15 concentrations.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from .figureCommon import subplotLabel, getSetup, global_legend, calc_dose_response, catplot_comparison, nllsq_EC50
from .figureS5 import plot_exp_v_pred
from ..imports import import_pstat, import_Rexpr, import_samples_2_15

ckineConc, cell_names_pstat, IL2_data, IL2_data2, IL15_data, IL15_data2 = import_pstat(combine_samples=False)
_, _, IL2_data_avg, IL15_data_avg, _ = import_pstat(combine_samples=True)
unkVec_2_15_glob = import_samples_2_15(N=1)  # use one rate
_, receptor_data, cell_names_receptor = import_Rexpr()

pstat_data = {
    "Experiment 1": np.concatenate((IL2_data.astype(np.float), IL15_data.astype(np.float)), axis=None),
    "Experiment 2": np.concatenate((IL2_data2.astype(np.float), IL15_data2.astype(np.float)), axis=None),
    "IL": np.concatenate(((np.tile(np.array("IL-2"), len(cell_names_pstat) * 4 * len(ckineConc))), np.tile(np.array("IL-15"), len(cell_names_pstat) * 4 * len(ckineConc))), axis=None),
}
pstat_df = pd.DataFrame(data=pstat_data)


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))

    subplotLabel(ax)

    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    tpsSC = np.array([0.5, 1.0]) * 60.0
    compare_experimental_data(ax[0], pstat_df)  # compare experiment 1 to 2
    df = WT_EC50s(unkVec_2_15_glob)
    catplot_comparison(ax[1], df, Mut=False)  # compare experiments to model predictions
    plot_corrcoef(ax[2], tps, unkVec_2_15_glob)  # find correlation coefficients
    global_legend(ax[0], Mut=True, exppred=False)  # add legend subplots A-C

    plot_exp_v_pred(ax[3:9], tpsSC, cell_subset=["NK", "CD8+", "T-reg"])  # NK, CD8+, and Treg subplots taken from fig S5

    return f


def compare_experimental_data(ax, df):
    """ Compare both pSTAT5 replicates. """
    df.dropna(axis=0, how="any", inplace=True)
    df["Experiment 1"] /= 100.0
    df["Experiment 2"] /= 100.0
    sns.set_palette(sns.xkcd_palette(["violet", "goldenrod"]))
    sns.scatterplot(x="Experiment 1", y="Experiment 2", hue="IL", data=df, ax=ax, s=10, legend=False)
    ax.set_aspect("equal", "box")


def plot_corrcoef(ax, tps, unkVec_2_15, Traf=True):
    """ Plot correlation coefficients between predicted and experimental data for all cell types. """
    corr_coefs = np.zeros(2 * len(cell_names_receptor))

    pred_data2, pred_data15 = calc_dose_response(cell_names_receptor, unkVec_2_15, receptor_data, tps, ckineConc, IL2_data_avg, IL15_data_avg, Traf)

    for l, _ in enumerate(cell_names_receptor):
        corr_coef2 = pearsonr(IL2_data_avg[(l * 4): ((l + 1) * 4)].flatten(), np.squeeze(pred_data2[l, :, :, :]).T.flatten())
        corr_coef15 = pearsonr(IL15_data_avg[(l * 4): ((l + 1) * 4)].flatten(), np.squeeze(pred_data15[l, :, :, :]).T.flatten())
        corr_coefs[l] = corr_coef2[0]
        corr_coefs[len(cell_names_receptor) + l] = corr_coef15[0]

    x_pos = np.arange(len(cell_names_receptor))
    ax.bar(x_pos - 0.15, corr_coefs[0: len(cell_names_receptor)], width=0.3, color="darkorchid", label="IL2", tick_label=cell_names_receptor)
    ax.bar(x_pos + 0.15, corr_coefs[len(cell_names_receptor): (2 * len(cell_names_receptor))], width=0.3, color="goldenrod", label="IL15", tick_label=cell_names_receptor)
    ax.set(ylabel=("Correlation"), ylim=(0.0, 1.0))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontsize=6.8, rotation_mode="anchor", ha="right")


def calculate_predicted_EC50(x0, receptors, tps, cell_index, unkVec_2_15, Traf=True):
    """ Calculate average EC50 from model predictions. """
    IL2_activity, IL15_activity = calc_dose_response(cell_names_pstat, unkVec_2_15, receptors, tps, ckineConc, IL2_data_avg, IL15_data_avg, Traf)
    EC50_2 = np.zeros(len(tps))
    EC50_15 = EC50_2.copy()
    # calculate EC50 for each timepoint... using 0 in activity matrices since we only have 1 sample from unkVec_2_15
    for i, _ in enumerate(tps):
        EC50_2[i] = nllsq_EC50(x0, np.log10(ckineConc.astype(np.float) * 10 ** 4), IL2_activity[cell_index, :, 0, i])
        EC50_15[i] = nllsq_EC50(x0, np.log10(ckineConc.astype(np.float) * 10 ** 4), IL15_activity[cell_index, :, 0, i])
    return EC50_2, EC50_15


def WT_EC50s(unkVec_2_15, Traf=True):
    """Returns dataframe of the Wild Type EC50s"""
    df = pd.DataFrame(columns=["Time Point", "Cell Type", "IL", "Data Type", "EC50"])

    x0 = [1, 2.0, 1000.0]
    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    data_types = []
    cell_types = []
    EC50s_2 = np.zeros(len(cell_names_pstat) * len(tps) * 2)
    EC50s_15 = np.zeros(len(cell_names_pstat) * len(tps) * 2)

    for i, name in enumerate(cell_names_pstat):
        assert cell_names_pstat[i] == cell_names_receptor[i]
        celltype_data_2 = IL2_data_avg[(i * 4): ((i + 1) * 4)]
        celltype_data_15 = IL15_data_avg[(i * 4): ((i + 1) * 4)]
        data_types.append(np.tile(np.array("Predicted"), len(tps)))
        # predicted EC50
        EC50_2, EC50_15 = calculate_predicted_EC50(x0, receptor_data, tps, i, unkVec_2_15, Traf)
        for j, item in enumerate(EC50_2):
            EC50s_2[(2 * len(tps) * i) + j] = item
            EC50s_15[(2 * len(tps) * i) + j] = EC50_15[j]
        # experimental EC50
        for k, _ in enumerate(tps):
            timepoint_data_2 = celltype_data_2[k]
            timepoint_data_15 = celltype_data_15[k]
            EC50s_2[len(tps) + (2 * len(tps) * i) + k] = nllsq_EC50(x0, np.log10(ckineConc.astype(np.float) * 10 ** 4), timepoint_data_2)
            EC50s_15[len(tps) + (2 * len(tps) * i) + k] = nllsq_EC50(x0, np.log10(ckineConc.astype(np.float) * 10 ** 4), timepoint_data_15)
        data_types.append(np.tile(np.array("Experimental"), len(tps)))
        cell_types.append(np.tile(np.array(name), len(tps) * 2))  # for both experimental and predicted

    EC50 = np.concatenate((EC50s_2, EC50s_15), axis=None)
    EC50 = EC50 - 4  # account for 10^4 multiplication
    data_types = np.tile(np.array(data_types).reshape(80), 2)  # for IL2 and IL15
    cell_types = np.tile(np.array(cell_types).reshape(80), 2)
    IL = np.concatenate((np.tile(np.array("IL-2"), len(cell_names_pstat) * len(tps) * 2), np.tile(np.array("IL-15"), len(cell_names_pstat) * len(tps) * 2)), axis=None)
    data = {"Time Point": np.tile(np.array(tps), len(cell_names_pstat) * 4), "IL": IL, "Cell Type": cell_types.reshape(160), "Data Type": data_types.reshape(160), "EC-50": EC50}
    df = pd.DataFrame(data)
    return df
