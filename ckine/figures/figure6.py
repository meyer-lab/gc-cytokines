"""
This creates Figure 6.
"""
import string
import numpy as np
import pandas as pd
import seaborn as sns
import theano.tensor as T
import theano
from .figureCommon import subplotLabel, getSetup, global_legend, calc_dose_response, grouped_scaling
from ..imports import import_pstat, import_samples_2_15, import_Rexpr
from ..model import getTotalActiveSpecies, receptor_expression, getRateVec, getparamsdict, runCkineUP
from ..differencing_op import runCkineDoseOp

unkVec_2_15, scales = import_samples_2_15(N=1)
_, receptor_data, cell_names_receptor = import_Rexpr()
ckineConc, cell_names_pstat, IL2_data, IL15_data, _ = import_pstat()
ckineC = ckineConc[7]
time = 240.
unkVec = getRateVec(unkVec_2_15)
CondIL = np.zeros((1, 6), dtype=np.float64)
CondIL[0, 0] = ckineC
Op = runCkineDoseOp(tt=np.array(time), condense=getTotalActiveSpecies().astype(np.float64), conditions=CondIL)
unkVec = unkVec[6::].flatten()
unkVecT = T.set_subtensor(T.zeros(54)[0:], np.transpose(unkVec))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7.5, 6), (3, 4), multz={4: 3, 8: 3})

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    ckines = ['IL-2', 'IL-15']
    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0

    df_spec = pd.DataFrame(columns=['Cells', 'Ligand', 'Time', 'Concentration', 'Data Type', 'Specificity'])
    df_act = pd.DataFrame(columns=['Cells', 'Ligand', 'Time', 'Concentration', 'Activity Type', 'Activity'])

    IL2_activity, IL15_activity = calc_dose_response(cell_names_pstat, unkVec_2_15, scales, receptor_data, tps, ckineConc, IL2_data, IL15_data)
    for i, name in enumerate(cell_names_pstat):
        assert cell_names_pstat[i] == cell_names_receptor[i]
        df_add2 = pd.DataFrame({'Cells': np.tile(name, len(ckineConc) * len(tps) * 2), 'Ligand': np.tile('IL-2', len(ckineConc) * len(tps) * 2),
                                'Time': np.tile(np.repeat(tps, len(ckineConc)), 2), 'Concentration': np.tile(ckineConc, len(tps) * 2),
                                'Activity Type': np.concatenate((np.tile('Experimental', len(tps) * len(ckineConc)), np.tile('Predicted', len(tps) * len(ckineConc)))),
                                'Activity': np.concatenate((IL2_data[(i * 4): ((i + 1) * 4)].reshape(48,), np.squeeze(IL2_activity[i, :, :, :]).T.reshape(48,)))})
        df_add15 = pd.DataFrame({'Cells': np.tile(name, len(ckineConc) * len(tps) * 2), 'Ligand': np.tile('IL-15', len(ckineConc) * len(tps) * 2),
                                 'Time': np.tile(np.repeat(tps, len(ckineConc)), 2), 'Concentration': np.tile(ckineConc, len(tps) * 2),
                                 'Activity Type': np.concatenate((np.tile('Experimental', len(tps) * len(ckineConc)), np.tile('Predicted', len(tps) * len(ckineConc)))),
                                 'Activity': np.concatenate((IL15_data[(i * 4): ((i + 1) * 4)].reshape(48,), np.squeeze(IL15_activity[i, :, :, :]).T.reshape(48,)))})
        df_act = df_act.append(df_add2, ignore_index=True)
        df_act = df_act.append(df_add15, ignore_index=True)

    df_act.drop(df_act[(df_act.Cells == 'Naive Treg') | (df_act.Cells == 'Mem Treg') | (df_act.Cells == 'Naive Th') |
                       (df_act.Cells == 'Mem Th') | (df_act.Cells == 'Naive CD8+') | (df_act.Cells == 'Mem CD8+')].index, inplace=True)
    ckineConc_ = np.delete(ckineConc, 11, 0)  # delete smallest concentration since zero/negative activity

    calc_plot_specificity(ax, df_spec, df_act, ckines, ckineConc_)
    global_legend(ax[0])
    Specificity(ax=ax[4])
    Spec_Aff(ax[5], 40, unkVecT, scalesT)

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


def genscalesTh(scalesSc, unkVecSc, cytokCSc, expr_act2Sc, expr_act15Sc):
    """Function to generate scaling factors for predicted pSTAT levels"""
    _, receptor_dataSc, cell_names = import_Rexpr()
    tpsSc = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    rxntfr2 = unkVecSc.T.copy()
    total_activity2 = np.zeros((len(cell_names), cytokCSc.shape[0], rxntfr2.shape[0], tpsSc.size))
    total_activity15 = total_activity2.copy()

    for i, _ in enumerate(cell_names):
        # updates rxntfr for receptor expression for IL2Ra, IL2Rb, gc
        cell_data = receptor_dataSc[i]
        rxntfr2[:, 22] = receptor_expression(cell_data[0], rxntfr2[:, 17], rxntfr2[:, 20], rxntfr2[:, 19], rxntfr2[:, 21])
        rxntfr2[:, 23] = receptor_expression(cell_data[1], rxntfr2[:, 17], rxntfr2[:, 20], rxntfr2[:, 19], rxntfr2[:, 21])
        rxntfr2[:, 24] = receptor_expression(cell_data[2], rxntfr2[:, 17], rxntfr2[:, 20], rxntfr2[:, 19], rxntfr2[:, 21])
        rxntfr2[:, 25] = 0.0  # We never observed any IL-15Ra

        rxntfr15 = rxntfr2.copy()

        # loop for each IL2 concentration
        for j in range(cytokCSc.shape[0]):
            rxntfr2[:, 0] = rxntfr15[:, 1] = cytokCSc[1]  # assign concs for each cytokine

            # handle case of IL-2
            yOut = runCkineUP(tpsSc, rxntfr2)
            activity2 = np.dot(yOut, getTotalActiveSpecies().astype(np.float))
            # handle case of IL-15
            yOut = runCkineUP(tpsSc, rxntfr15)
            activity15 = np.dot(yOut, getTotalActiveSpecies().astype(np.float))

            total_activity2[i, j, :, :] = np.reshape(activity2, (-1, 4))  # save the activity from this concentration for all 4 tps
            total_activity15[i, j, :, :] = np.reshape(activity15, (-1, 4))  # save the activity from this concentration for all 4 tps
    scaleTh = grouped_scaling(scalesSc, cell_names, expr_act2Sc, expr_act15Sc, total_activity2, total_activity15)
    return scaleTh


scalesT = genscalesTh(scales, unkVec_2_15, ckineConc, IL2_data, IL15_data)


def Specificity(ax):
    """ Creates Theano Function for calculating Specificity gradient with respect to various parameters"""
    S = OPgenSpec(unkVecT, scalesT)
    Sgrad = T.grad(S[0], unkVecT)
    Sgradfunc = theano.function([unkVecT], Sgrad)
    Sfunc = theano.function([unkVecT], S[0])

    S_partials = Sgradfunc(unkVec.flatten()) / Sfunc(unkVec.flatten())

    names = list(getparamsdict(np.zeros(60)).keys())[6::]
    df = pd.DataFrame(data={'rate': names, 'value': S_partials})

    df.drop(df.index[-8:], inplace=True)
    df.drop(df.index[7:21], inplace=True)
    df.drop(df.index[13:27], inplace=True)
    df.drop(df.index[0], inplace=True)
    df.drop(df.index[-5:], inplace=True)

    sns.barplot(data=df, x='rate', y='value', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, rotation_mode="anchor", ha="right")
    ax.set_ylim(-0.25, 0.25)


def OPgen(unkVecOP, CellTypes, OpC, scalesTh, RaAffM, RbAffM):
    "Generates the UnkVec with cell specific receptor abundances and expression rates"
    _, receptor_dataC, cell_names_receptorC = import_Rexpr()
    cell_names_receptorC = cell_names_receptorC.tolist()

    # set true Op
    cell_data = receptor_dataC[cell_names_receptorC.index(CellTypes), :]
    unkVecOP = T.set_subtensor(unkVecOP[46], receptor_expression(cell_data[0], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # RA
    unkVecOP = T.set_subtensor(unkVecOP[47], receptor_expression(cell_data[1], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Rb
    unkVecOP = T.set_subtensor(unkVecOP[48], receptor_expression(cell_data[2], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Gc
    unkVecOP = T.set_subtensor(unkVecOP[49], 0)

    # Adjust a affinities
    for ii in [1, 21]:
        unkVecOP = T.set_subtensor(unkVecOP[ii], unkVecOP[ii] * RaAffM)

    # Adjust b/g affinities
    for ii in [2, 3, 4, 5, 6, 22, 23, 24, 25, 26]:
        unkVecOP = T.set_subtensor(unkVecOP[ii], unkVecOP[ii] * RbAffM)

    cell_groups = np.array([['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+', 'Naive CD8+', 'Mem CD8+']])
    for i, group in enumerate(cell_groups):
        group = np.array(group)
        if np.where(group == CellTypes)[0].size > 0:
            scale1 = scalesTh[i, 1, 0]
            scale2 = scalesTh[i, 0, 0]

    Cell_Op = (OpC(unkVecOP) * scale1) / (OpC(unkVecOP) + scale2)

    return Cell_Op


def OPgenSpec(unk, scalesIn, k1Aff=1.0, k5Aff=1.0):
    """ Make an Op for specificity from the given conditions. """
    S = (OPgen(unk, "T-reg", Op, scalesIn, k1Aff, k5Aff) /
         (OPgen(unk, "T-reg", Op, scalesIn, k1Aff, k5Aff) +
          OPgen(unk, "T-helper", Op, scalesIn, k1Aff, k5Aff) +
          OPgen(unk, "NK", Op, scalesIn, k1Aff, k5Aff) +
          OPgen(unk, "CD8+", Op, scalesIn, k1Aff, k5Aff)))

    return S


def Spec_Aff(ax, npoints, unkVecAff, scalesAff):
    "Plots specificity for a cell type over a range of IL2RBG and IL2Ra affinities"
    affRange = np.logspace(2, -1, npoints)
    RaAff = np.array([1, 10])
    specHolder = np.zeros([len(RaAff), npoints])
    for i, k1Aff in enumerate(RaAff):
        for j, k5Aff in enumerate(affRange):
            specHolder[i, j] = OPgenSpec(unkVecAff, scalesAff, k1Aff, k5Aff).eval()
        ax.plot(1 / affRange, specHolder[i, :], label=str(1 / RaAff[i]) + " IL2Ra Affinity")

    ax.set_xscale('log')
    ax.set_xlabel('Relative CD122/CD132 Affinity')
    ax.set_ylabel('T-reg Specificity')
    ax.legend()
