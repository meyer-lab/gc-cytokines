"""
This creates Figure S7.
"""
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, plot_conf_int, import_pMuteins, organize_expr_pred, mutein_scaling
from ..imports import import_Rexpr, import_samples_2_15, import_pstat

dataMean = import_pMuteins()
dataMean.reset_index(inplace=True)
data, _, _ = import_Rexpr()
data.reset_index(inplace=True)
unkVec_2_15, _ = import_samples_2_15(N=25)
_, _, _, _, pstat_df = import_pstat()
dataMean = dataMean.append(pstat_df, ignore_index=True, sort=True)


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 18), (8, 6))

    subplotLabel(ax)

    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    muteinC = dataMean.Concentration.unique()
    dataMean["Concentration"] = dataMean["Concentration"].astype(np.float)  # logscale for plotting

    ligand_order = ['WT N-term', 'WT C-term', 'V91K C-term', 'R38Q N-term', 'F42Q N-Term', 'N88D C-term']
    cell_order = ['NK', 'CD8+', 'T-reg', 'Naive Treg', 'Mem Treg', 'T-helper', 'Naive Th', 'Mem Th']

    df = pd.DataFrame(columns=['Cells', 'Ligand', 'Time Point', 'Concentration', 'Activity Type', 'Replicate', 'Activity'])  # make empty dataframe for all cell types

    # loop for each cell type and mutein
    for _, cell_name in enumerate(cell_order):

        IL2Ra = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == 'IL-2R$\\alpha$'), "Count"].values[0]
        IL2Rb = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == 'IL-2R$\\beta$'), "Count"].values[0]
        gc = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == '$\\gamma_{c}$'), "Count"].values[0]
        receptors = np.array([IL2Ra, IL2Rb, gc]).astype(np.float)

        for _, ligand_name in enumerate(ligand_order):

            # append dataframe with experimental and predicted activity
            df = organize_expr_pred(df, cell_name, ligand_name, receptors, muteinC, tps, unkVec_2_15)

    # determine scaling constants
    scales = mutein_scaling(df, unkVec_2_15)
    plot_expr_predM(ax, df, scales, cell_order, ligand_order, tps, muteinC)

    return f


def plot_expr_predM(ax, df, scales, cell_order, ligand_order, tps, muteinC):
    """ Plots experimental and scaled model-predicted dose response for all cell types, muteins, and time points. """

    pred_data = np.zeros((12, 4, unkVec_2_15.shape[1]))
    cell_groups = [['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+']]
    ylims = [50000., 30000., 2500., 3500.]

    for i, cell_name in enumerate(cell_order):
        for j, ligand_name in enumerate(ligand_order):
            axis = i * 8 + j

            # plot experimental data
            sns.scatterplot(x="Concentration", y="RFU", hue="Time", data=dataMean.loc[(dataMean["Cells"] == cell_name)
                                                                                      & (dataMean["Ligand"] == ligand_name)], ax=ax[axis], s=10, palette=cm.rainbow, legend=False)

            # scale and plot model predictions
            for k, conc in enumerate(df.Concentration.unique()):
                for l, tp in enumerate(tps):
                    for m in range(unkVec_2_15.shape[1]):
                        pred_data[k, l, m] = df.loc[(df["Cells"] == cell_name) & (df["Ligand"] == ligand_name) & (
                            df["Activity Type"] == 'predicted') & (df["Concentration"] == conc) & (df["Time Point"] == tp) & (df["Replicate"] == (m + 1)), "Activity"]

            for n, cell_names in enumerate(cell_groups):
                if cell_name in cell_names:
                    for o in range(unkVec_2_15.shape[1]):
                        pred_data[:, :, o] = scales[n, 1, o] * pred_data[:, :, o] / (pred_data[:, :, o] + scales[n, 0, o])

                    plot_dose_responseM(ax[axis], pred_data, tps, muteinC, legend=(axis == 0))
                    ax[axis].set(ylim=(0, ylims[n]))
            ax[axis].set(xlabel=("[" + ligand_name + "] Concentration (nM)"), ylabel="Activity", title=cell_name)


def plot_dose_responseM(ax, mutein_activity, tps, muteinC, legend=False):
    """ Plots predicted activity for multiple timepoints and mutein concentrations. """
    colors = cm.rainbow(np.linspace(0, 1, tps.size))

    for tt in range(tps.size):
        if not legend:
            plot_conf_int(ax, muteinC.astype(np.float), mutein_activity[:, tt, :], colors[tt])
        else:
            plot_conf_int(ax, muteinC.astype(np.float), mutein_activity[:, tt, :], colors[tt], (tps[tt] / 60.0).astype(str))
            ax.legend(title="time (hours)")
    ax.set_xscale('log')
    ax.set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
