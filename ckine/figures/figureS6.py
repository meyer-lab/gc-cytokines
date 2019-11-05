"""
This creates Figure S6.
"""
import string
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import least_squares
from .figureCommon import subplotLabel, getSetup, global_legend
from ..model import receptor_expression, runCkineU, getTotalActiveCytokine
from ..imports import import_pstat, import_samples_2_15, import_Rexpr

_, _, _, _, dataMean = import_pstat()
dataMean.reset_index(inplace=True)
unkVec, _ = import_samples_2_15(N=1)
Rexpr, _, _ = import_Rexpr()


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (1, 1))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    ckines = ['IL2', 'IL15']
    conc = dataMean.Concentration.unique()

    df_spec = pd.DataFrame(columns=['Cells', 'Ligand', 'Time', 'Concentration', 'Data Type', 'Specificity'])
    df_act = pd.DataFrame(columns=['Cells', 'Ligand', 'Time', 'Concentration', 'Activity Type', 'Activity'])
    dataMean.insert(5, "Activity Type", np.tile(np.array('Experimental'), 960))  # add activity type column so can append to df_act

    df_act = calc_scaled_activity(df_act, ckines, conc)
    df_act.drop(df_act[df_act.Cells == 'Naive Treg'].index, inplace=True)
    df_act.drop(df_act[df_act.Cells == 'Mem Treg'].index, inplace=True)  # delete Naive/Mem Treg so only comparing to non-Treg cells
    conc = np.delete(conc, 11, 0)  # delete smallest concentration since zero/negative activity
    calc_plot_specificity(ax, df_spec, df_act, ckines, conc)
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
                                (df_activity["Concentration"] == concentration) & (df_activity["Activity Type"] == dtype), "RFU"].values[0]
        pstat_sum = 0.
        for cell in df_activity.Cells.unique():
            pstat_ = df_activity.loc[(df_activity["Cells"] == cell) & (df_activity["Ligand"] == ligand) & (df_activity["Time"] == tp) &
                                     (df_activity["Concentration"] == concentration) & (df_activity["Activity Type"] == dtype), "RFU"].values[0]
            pstat_sum = pstat_sum + pstat_
        df_add = pd.DataFrame({'Cells': cell_type, 'Ligand': ligand, 'Time': tp, 'Concentration': concentration, 'Data Type': dtype, 'Specificity': pstat / pstat_sum}, index=[0])
        df_specificity = df_specificity.append(df_add, ignore_index=True)

    return df_specificity


def calc_scaled_activity(df_activity, ligands, concs):
    """ Calculates scaled dose response for all cell populations. """

    def organize_pred(df, cell_name, ligand_name, receptors, concs, tps):
        """ Appends input dataframe with predicted activity for a given cell type and ligand. """

        num = tps.size * len(concs)

        # calculate predicted dose response
        pred_data = np.zeros((12, 1))
        cell_receptors = receptor_expression(receptors, unkVec[17], unkVec[20], unkVec[19], unkVec[21])
        pred_data[:, :] = calc_dose_response(unkVec, tps, concs, ligand_name, cell_receptors)
        df_pred = pd.DataFrame({'Cells': np.tile(np.array(cell_name), num), 'Ligand': np.tile(np.array(ligand_name), num), 'Time': np.tile(
            tps, num), 'Concentration': concs, 'Activity Type': np.tile(np.array('Predicted'), num), 'Activity': pred_data[:, :].reshape(num,)})
        df = df.append(df_pred, ignore_index=True)

        return df

    for _, ligand in enumerate(ligands):
        for _, cell_type in enumerate(dataMean.Cells.unique()):
            IL2Ra = Rexpr.loc[(Rexpr["Cell Type"] == cell_type) & (Rexpr["Receptor"] == 'IL-2R$\\alpha$'), "Count"].values[0]
            IL2Rb = Rexpr.loc[(Rexpr["Cell Type"] == cell_type) & (Rexpr["Receptor"] == 'IL-2R$\\beta$'), "Count"].values[0]
            gc = Rexpr.loc[(Rexpr["Cell Type"] == cell_type) & (Rexpr["Receptor"] == '$\\gamma_{c}$'), "Count"].values[0]
            receptors = np.array([IL2Ra, IL2Rb, gc]).astype(np.float)
            df_activity = organize_pred(df_activity, cell_type, ligand, receptors, concs, np.array(60.))

    # scale predictions
    df_activity = df_activity.append(dataMean.loc[(dataMean["Time"] == 60.)], ignore_index=True)
    scales = activity_scaling(df_activity)
    cell_groups = [['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+', 'Naive CD8+', 'Mem CD8+']]
    for _, cell_name in enumerate(df_activity.Cells.unique()):
        for i, cell_names in enumerate(cell_groups):
            if cell_name in cell_names:
                df_activity.loc[(df_activity['Activity Type'] == 'Predicted') & (df_activity['Cells'] == cell_name), 'RFU'] = (scales[i, 1]
                                                                                                                               * df_activity.loc[(df_activity['Activity Type'] == 'Predicted') & (df_activity['Cells'] == cell_name), 'Activity']) \
                    / (df_activity.loc[(df_activity['Activity Type'] == 'Predicted') & (df_activity['Cells'] == cell_name), 'Activity'] + scales[i, 0])

    return df_activity


def calc_dose_response(unkvec, tps, concs, ligand_name, cell_receptors):
    """ Calculates activity for a given cell type at various concentrations and timepoints. """

    total_activity = np.zeros((len(concs), tps.size))

    # loop for each concentration
    for i, conc in enumerate(concs):
        unkvec[0] = unkvec[1] = 0
        if ligand_name == 'IL15':
            unkvec[1] = conc
        elif ligand_name == 'IL2':
            unkvec[0] = conc
        unkvec[22:25] = cell_receptors.reshape(3, 1)  # set receptor expression for IL2Ra, IL2Rb, gc
        unkvec[25] = 0.0  # we never observed any IL-15Ra
        yOut = runCkineU(tps, unkvec.T)
        active_ckine = np.zeros(yOut.shape[0])
        for j in range(yOut.shape[0]):
            if ligand_name == 'IL15':
                active_ckine[j] = getTotalActiveCytokine(1, yOut[j, :])
            elif ligand_name == 'IL2':
                active_ckine[j] = getTotalActiveCytokine(0, yOut[j, :])
        total_activity[i, :] = np.reshape(active_ckine, (-1, 1))  # save the activity from this concentration for all 4 tps

    return total_activity


def activity_scaling(df):
    """ Determines scaling parameters for specified cell groups for across all ligands. """

    cell_groups = [['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+', 'Naive CD8+', 'Mem CD8+']]

    scales = np.zeros((4, 2))
    for i, cells in enumerate(cell_groups):
        subset_df = df[df['Cells'].isin(cells)]
        scales[i, :] = optimize_scale(np.array(subset_df.loc[(subset_df["Activity Type"] == 'Predicted'), "Activity"]),
                                      np.array(subset_df.loc[(subset_df["Activity Type"] == 'Experimental'), "RFU"]))

    return scales


def optimize_scale(model_act, exp_act):
    """ Formulates the optimal scale to minimize the residual between model activity predictions and experimental activity measurments for a given cell type. """

    # scaling factors are sigmoidal and linear, respectively
    guess = np.array([100.0, np.mean(exp_act) / np.mean(model_act)])

    def calc_res(sc):
        """ Calculate the residuals. This is the function we minimize. """
        scaled_act = sc[1] * model_act / (model_act + sc[0])
        err = exp_act - scaled_act
        return err.flatten()

    # find result of minimization where both params are >= 0
    res = least_squares(calc_res, guess, bounds=(0.0, np.inf))
    return res.x
