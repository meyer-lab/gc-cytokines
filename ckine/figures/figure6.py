"""
This creates Figure S6.
"""
import string
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import least_squares
from .figureCommon import subplotLabel, getSetup, global_legend
from .figureS5 import calc_dose_response
from ..model import receptor_expression, runCkineU, getTotalActiveCytokine
from ..imports import import_pstat, import_samples_2_15, import_Rexpr

unkVec_2_15, scales = import_samples_2_15(N=1)
_, receptor_data, cell_names_receptor = import_Rexpr()
ckineConc, cell_names_pstat, IL2_data, IL15_data, _ = import_pstat()


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (1, 1))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    ckines = ['IL-2', 'IL-15']
    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0

    df_spec = pd.DataFrame(columns=['Cells', 'Ligand', 'Time', 'Concentration', 'Data Type', 'Specificity'])
    df_act = pd.DataFrame(columns=['Cells', 'Ligand', 'Time', 'Concentration', 'Activity Type', 'Activity'])
    
    for i, name in enumerate(cell_names_pstat):
        assert cell_names_pstat[i] == cell_names_receptor[i]
        IL2_activity, IL15_activity = calc_dose_response(unkVec_2_15, scales, receptor_data[i], tps, ckineConc, IL2_data[(i * 4): ((i + 1) * 4)], IL15_data[(i * 4): ((i + 1) * 4)])
        df_add2 = pd.DataFrame({'Cells': np.tile(name, len(ckineConc) * len(tps) * 2), 'Ligand': np.tile('IL-2', len(ckineConc) * len(tps) * 2), 'Time': np.tile(np.repeat(tps, len(ckineConc)), 2), 'Concentration': np.tile(ckineConc, len(tps) * 2), 'Activity Type': np.concatenate((np.tile('Experimental', len(tps) * len(ckineConc)), np.tile('Predicted', len(tps) * len(ckineConc)))), 'Activity': np.concatenate((IL2_data[(i * 4): ((i + 1) * 4)].reshape(48,), np.squeeze(IL2_activity).T.reshape(48,)))})
        df_add15 = pd.DataFrame({'Cells': np.tile(name, len(ckineConc) * len(tps) * 2), 'Ligand': np.tile('IL-15', len(ckineConc) * len(tps) * 2), 'Time': np.tile(np.repeat(tps, len(ckineConc)), 2), 'Concentration': np.tile(ckineConc, len(tps) * 2), 'Activity Type': np.concatenate((np.tile('Experimental', len(tps) * len(ckineConc)), np.tile('Predicted', len(tps) * len(ckineConc)))), 'Activity': np.concatenate((IL15_data[(i * 4): ((i + 1) * 4)].reshape(48,), np.squeeze(IL15_activity).T.reshape(48,)))})
        df_act = df_act.append(df_add2, ignore_index=True)
        df_act = df_act.append(df_add15, ignore_index=True)

    df_act.drop(df_act[df_act.Cells == 'Naive Treg'].index, inplace=True)
    df_act.drop(df_act[df_act.Cells == 'Mem Treg'].index, inplace=True)  # delete Naive/Mem Treg so only comparing to non-Treg cells
    ckineConc_ = np.delete(ckineConc, 11, 0)  # delete smallest concentration since zero/negative activity
    calc_plot_specificity(ax, df_spec, df_act, ckines, ckineConc_)
    global_legend(ax[0])

    return f


def calc_plot_specificity(ax, df_specificity, df_activity, ligands, concs):
    """ Calculates and plots specificity for both cytokines and experimental/predicted activity for T-regs. """

    # calculate specificity and append to dataframe
    for _, ckine in enumerate(ligands):
        for _, conc in enumerate(concs):
            df_specificity = specificity(df_specificity, df_activity, 'T-reg', ckine, 60., conc)

    df_specificity["Concentration"] = np.log10(df_specificity["Concentration"].astype(np.float))  # logscale for plotting

    # plot all specificty values
    sns.set_palette(sns.xkcd_palette(["violet", "goldenrod"]))
    sns.scatterplot(x="Concentration", y="Specificity", hue="Ligand", data=df_specificity.loc[(df_specificity["Cells"] ==
                                                                                               'T-reg') & (df_specificity["Data Type"] == 'Experimental')], ax=ax[0], marker='o', legend=False)
    sns.scatterplot(x="Concentration", y="Specificity", hue="Ligand", data=df_specificity.loc[(df_specificity["Cells"] == 'T-reg') &
                                                                                              (df_specificity["Data Type"] == 'Predicted')], ax=ax[0], marker='^', legend=False)
    ax[0].set(xlabel="(log$_{10}$[nM])", ylabel="Specificity", ylim=[0, 1.], title='T-reg')


def specificity(df_specificity, df_activity, cell_type, ligand, tp, concentration):
    """ Caculate specificity value of cell type. """

    data_types = ['Experimental', 'Predicted']
    for _, dtype in enumerate(data_types):
        pstat = df_activity.loc[(df_activity["Cells"] == cell_type) & (df_activity["Ligand"] == ligand) & (df_activity["Time"] == tp) &
                                (df_activity["Concentration"] == concentration) & (df_activity["Activity Type"] == dtype), "Activity"].values[0]
        pstat_sum = 0.
        for cell in df_activity.Cells.unique():
            pstat_ = df_activity.loc[(df_activity["Cells"] == cell) & (df_activity["Ligand"] == ligand) & (df_activity["Time"] == tp) &
                                     (df_activity["Concentration"] == concentration) & (df_activity["Activity Type"] == dtype), "Activity"].values[0]
            pstat_sum = pstat_sum + pstat_
        df_add = pd.DataFrame({'Cells': cell_type, 'Ligand': ligand, 'Time': tp, 'Concentration': concentration, 'Data Type': dtype, 'Specificity': pstat / pstat_sum}, index=[0])
        df_specificity = df_specificity.append(df_add, ignore_index=True)

    return df_specificity
