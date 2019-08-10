"""
This creates Figure 4. Comparison of Experimental verus Predicted Activity across IL2 and IL15 concentrations.
"""

import string
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import least_squares
from scipy.stats import pearsonr
from .figureCommon import subplotLabel, getSetup
from .figureS5 import calc_dose_response
from ..imports import import_pstat, import_Rexpr, import_samples_2_15

ckineConc, cell_names_pstat, IL2_data, IL15_data = import_pstat()
unkVec_2_15, scales = import_samples_2_15(N=1)  # use one rate
_, receptor_data, cell_names_receptor = import_Rexpr()


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 3), (1, 2))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    df = pd.DataFrame(columns=['Time Point', 'Cell Type', 'IL', 'Data Type', 'EC50'])

    x0 = [1, 2., 1000.]
    tps = np.array([0.5, 1., 2., 4.]) * 60.
    data_types = []
    cell_types = []
    EC50s_2 = np.zeros(len(cell_names_pstat) * len(tps) * 2)
    EC50s_15 = np.zeros(len(cell_names_pstat) * len(tps) * 2)

    for i, name in enumerate(cell_names_pstat):
        assert cell_names_pstat[i] == cell_names_receptor[i]
        celltype_data_2 = IL2_data[(i * 4):((i + 1) * 4)]
        celltype_data_15 = IL15_data[(i * 4):((i + 1) * 4)]
        data_types.append(np.tile(np.array('Predicted'), len(tps)))
        # predicted EC50
        EC50_2, EC50_15 = calculate_predicted_EC50(x0, receptor_data[i], tps, celltype_data_2, celltype_data_15)
        for j, item in enumerate(EC50_2):
            EC50s_2[(2 * len(tps) * i) + j] = item
            EC50s_15[(2 * len(tps) * i) + j] = EC50_15[j]
        # experimental EC50
        for k, _ in enumerate(tps):
            timepoint_data_2 = celltype_data_2[k]
            timepoint_data_15 = celltype_data_15[k]
            EC50s_2[len(tps) + (2 * len(tps) * i) + k] = nllsq_EC50(x0, np.log10(ckineConc.astype(np.float) * 10**4), timepoint_data_2)
            EC50s_15[len(tps) + (2 * len(tps) * i) + k] = nllsq_EC50(x0, np.log10(ckineConc.astype(np.float) * 10**4), timepoint_data_15)
        data_types.append(np.tile(np.array('Experimental'), len(tps)))
        cell_types.append(np.tile(np.array(name), len(tps) * 2))  # for both experimental and predicted

    EC50 = np.concatenate((EC50s_2, EC50s_15), axis=None)
    EC50 = EC50 - 4  # account for 10^4 multiplication
    data_types = np.tile(np.array(data_types).reshape(80,), 2)  # for IL2 and IL15
    cell_types = np.tile(np.array(cell_types).reshape(80,), 2)
    IL = np.concatenate((np.tile(np.array('IL-2'), len(cell_names_pstat) * len(tps) * 2), np.tile(np.array('IL-15'), len(cell_names_pstat) * len(tps) * 2)), axis=None)
    data = {'Time Point': np.tile(np.array(tps), len(cell_names_pstat) * 4), 'IL': IL, 'Cell Type': cell_types.reshape(160,), 'Data Type': data_types.reshape(160,), 'EC-50': EC50}
    df = pd.DataFrame(data)

    catplot_comparison(ax[0], df)
    plot_corrcoef(ax[1], df, cell_names_pstat)

    return f


def catplot_comparison(ax, df):
    """ Construct EC50 catplots for each time point for IL2 and IL15. """
    # set a manual color palette
    col_list = ["violet", "goldenrod"]
    col_list_palette = sns.xkcd_palette(col_list)
    sns.set_palette(col_list_palette)
    # plot predicted EC50
    sns.catplot(x="Cell Type", y="EC-50", hue="IL",
                data=df.loc[(df['Time Point'] == 60.) & (df["Data Type"] == 'Predicted')],
                legend=False, legend_out=False, ax=ax, marker='o')
    # plot experimental EC50
    sns.catplot(x="Cell Type", y="EC-50", hue="IL",
                data=df.loc[(df['Time Point'] == 60.) & (df["Data Type"] == 'Experimental')],
                legend=False, legend_out=False, ax=ax, marker='^')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, rotation_mode="anchor", ha="right", position=(0, 0.02))
    ax.set_xlabel("")  # remove "Cell Type" from xlabel
    ax.set_ylabel(r"EC-50 (log$_{10}$[nM])")


def plot_corrcoef(ax, df, cell_types):
    """ Plot correlation coefficients between predicted and experimental data for all cell types. """
    corr_coefs = np.zeros(2 * len(cell_types))
    ILs = np.array(['IL-2', 'IL-15'])
    for i, name in enumerate(cell_types):
        for j, IL in enumerate(ILs):
            experimental_data = np.array(df.loc[(df['Data Type'] == 'Experimental') & (df['Cell Type'] == name) & (df['IL'] == IL), "EC-50"])
            predicted_data = np.array(df.loc[(df['Data Type'] == 'Predicted') & (df['Cell Type'] == name) & (df['IL'] == IL), "EC-50"])
            corr_coef = pearsonr(experimental_data, predicted_data)
            corr_coefs[j * len(cell_types) + i] = corr_coef[0]

    x_pos = np.arange(len(cell_types))
    ax.bar(x_pos - 0.15, corr_coefs[0:len(cell_types)], width=0.3, color='darkorchid', label='IL2', tick_label=cell_types)
    ax.bar(x_pos + 0.15, corr_coefs[len(cell_types):(2 * len(cell_types))], width=0.3, color='goldenrod', label='IL15', tick_label=cell_types)
    ax.set(ylabel=("Correlation Coefficient"), ylim=(0., 1.))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, rotation_mode="anchor", ha="right", position=(0, 0.02))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")


def calculate_predicted_EC50(x0, cell_receptor_data, tps, IL2_pstat, IL15_pstat):
    """ Calculate average EC50 from model predictions. """
    IL2_activity, IL15_activity = calc_dose_response(unkVec_2_15, scales, cell_receptor_data, tps, ckineConc, IL2_pstat, IL15_pstat)
    EC50_2 = np.zeros(len(tps))
    EC50_15 = EC50_2.copy()
    # calculate EC50 for each timepoint... using 0 in activity matrices since we only have 1 sample from unkVec_2_15
    for i, _ in enumerate(tps):
        EC50_2[i] = nllsq_EC50(x0, np.log10(ckineConc.astype(np.float) * 10**4), IL2_activity[:, 0, i])
        EC50_15[i] = nllsq_EC50(x0, np.log10(ckineConc.astype(np.float) * 10**4), IL15_activity[:, 0, i])
    return EC50_2, EC50_15


def nllsq_EC50(x0, xdata, ydata):
    """ Performs nonlinear least squares on activity measurements to determine parameters of Hill equation and outputs EC50. """
    lsq_res = least_squares(residuals, x0, args=(xdata, ydata), bounds=([0., 0., 0.], [10., 10., 10**5.]), jac='3-point')
    return lsq_res.x[0]


def hill_equation(x, x0, solution=0):
    """ Calculates EC50 from Hill Equation. """
    k = x0[0]
    n = x0[1]
    A = x0[2]
    xk = np.power(x / k, n)
    return (A * xk / (1.0 + xk)) - solution


def residuals(x0, x, y):
    """ Residual function for Hill Equation. """
    return hill_equation(x, x0) - y
