"""
This creates Figure 6.
"""
import string
import numpy as np
import pandas as pd
import seaborn as sns
import theano.tensor as T
import theano
from .figureCommon import subplotLabel, getSetup, global_legend, calc_dose_response
from ..imports import import_pstat, import_samples_2_15, import_Rexpr
from ..model import getTotalActiveSpecies, receptor_expression, getRateVec, getparamsdict
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

    IL2_activity, IL15_activity, _ = calc_dose_response(cell_names_pstat, unkVec_2_15, scales, receptor_data, tps, ckineConc, IL2_data, IL15_data)
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

    calc_plot_specificity(ax[0], 'NK', df_spec, df_act, ckines, ckineConc_)
    calc_plot_specificity(ax[1], 'T-helper', df_spec, df_act, ckines, ckineConc_)
    global_legend(ax[1])
    Specificity(ax=ax[4])
    Spec_Aff(ax[5], 40, unkVecT, scalesT)

    return f


def calc_plot_specificity(ax, cell_compare, df_specificity, df_activity, ligands, concs):
    """ Calculates and plots specificity for both cytokines and experimental/predicted activity for T-regs. """

    # calculate specificity and append to dataframe
    for _, ckine in enumerate(ligands):
        for _, conc in enumerate(concs):
            df_specificity = specificity(df_specificity, df_activity, 'T-reg', cell_compare, ckine, 60., conc)

    df_specificity["Concentration"] = np.log10(df_specificity["Concentration"].astype(np.float))  # logscale for plotting
    
    if cell_compare == 'T-helper':
        df_specificity.drop(df_specificity[(df_specificity['Cells'] == 'T-helper') & (df_specificity['Data Type'] == 'Experimental') & (df_specificity["Concentration"] < -2.5) & (df_specificity['Ligand'] == 'IL-2')].index, inplace=True)

    # plot all specificty values
    sns.set_palette(sns.xkcd_palette(["violet", "goldenrod"]))
    sns.scatterplot(x="Concentration", y="Specificity", hue="Ligand", data=df_specificity.loc[(df_specificity["Cells"] ==
                                                                                               cell_compare) & (df_specificity["Data Type"] == 'Experimental')], ax=ax, marker='o', legend=False)
    sns.scatterplot(x="Concentration", y="Specificity", hue="Ligand", data=df_specificity.loc[(df_specificity["Cells"] == cell_compare) &
                                                                                              (df_specificity["Data Type"] == 'Predicted')], ax=ax, marker='^', legend=False)
    ax.set(xlabel="(log$_{10}$[nM])", ylabel="Specificity", title=('T-reg vs. ' + cell_compare))


def specificity(df_specificity, df_activity, cell_type1, cell_type2, ligand, tp, concentration):
    """ Caculate specificity value of cell type. """

    data_types = ['Experimental', 'Predicted']
    for _, dtype in enumerate(data_types):
        pstat1 = df_activity.loc[(df_activity["Cells"] == cell_type1) & (df_activity["Ligand"] == ligand) & (df_activity["Time"] == tp) &
                                (df_activity["Concentration"] == concentration) & (df_activity["Activity Type"] == dtype), "Activity"].values[0]
        pstat2 = df_activity.loc[(df_activity["Cells"] == cell_type2) & (df_activity["Ligand"] == ligand) & (df_activity["Time"] == tp) &
                                (df_activity["Concentration"] == concentration) & (df_activity["Activity Type"] == dtype), "Activity"].values[0]
        df_add = pd.DataFrame({'Cells': cell_type2, 'Ligand': ligand, 'Time': tp, 'Concentration': concentration, 'Data Type': dtype, 'Specificity': pstat1 / pstat2}, index=[0])
        df_specificity = df_specificity.append(df_add, ignore_index=True)

    return df_specificity


tpsSc = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
_, _, scalesT = calc_dose_response(cell_names_receptor, unkVec_2_15, scales, receptor_data, tpsSc, ckineConc, IL2_data, IL15_data)


def Specificity(ax):
    """ Creates Theano Function for calculating Specificity gradient with respect to various parameters"""
    S_NK, S_Th = OPgenSpec(unkVecT, scalesT)
    SNKgrad = T.grad(S_NK[0], unkVecT)
    SThgrad = T.grad(S_Th[0], unkVecT)
    SNKgradfunc = theano.function([unkVecT], SNKgrad)
    SThgradfunc = theano.function([unkVecT], SThgrad)
    SNKfunc = theano.function([unkVecT], S_NK[0])
    SThfunc = theano.function([unkVecT], S_Th[0])

    SNK_partials = SNKgradfunc(unkVec.flatten()) / SNKfunc(unkVec.flatten())
    STh_partials = SThgradfunc(unkVec.flatten()) / SThfunc(unkVec.flatten())

    names = list(getparamsdict(np.zeros(60)).keys())[6::]
    dfNK = pd.DataFrame(data={'rate': names, 'value': SNK_partials})
    dfNK['cell'] = 'NK'
    dfTh = pd.DataFrame(data={'rate': names, 'value': STh_partials})
    dfTh['cell'] = 'T-Helper'

    dfNK.drop(dfNK.index[-8:], inplace=True)
    dfNK.drop(dfNK.index[7:21], inplace=True)
    dfNK.drop(dfNK.index[13:27], inplace=True)
    dfNK.drop(dfNK.index[0], inplace=True)
    dfNK.drop(dfNK.index[-5:], inplace=True)

    dfTh.drop(dfTh.index[-8:], inplace=True)
    dfTh.drop(dfTh.index[7:21], inplace=True)
    dfTh.drop(dfTh.index[13:27], inplace=True)
    dfTh.drop(dfTh.index[0], inplace=True)
    dfTh.drop(dfTh.index[-5:], inplace=True)

    df = pd.concat([dfNK, dfTh])

    sns.catplot(data=df, x='rate', y='value', kind="bar", hue='cell', ax=ax)
    ax.set_yscale("symlog", linthreshy=0.01)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, rotation_mode="anchor", ha="right")


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
    S_NK = (OPgen(unk, "T-reg", Op, scalesIn, k1Aff, k5Aff) /
            OPgen(unk, "NK", Op, scalesIn, k1Aff, k5Aff))

    S_Th = (OPgen(unk, "T-reg", Op, scalesIn, k1Aff, k5Aff) /
            OPgen(unk, "T-helper", Op, scalesIn, k1Aff, k5Aff))

    return S_NK, S_Th


def Spec_Aff(ax, npoints, unkVecAff, scalesAff):
    "Plots specificity for a cell type over a range of IL2RBG and IL2Ra affinities"
    affRange = np.logspace(2, -1, npoints)
    RaAff = np.array([1, 10])
    specHolderNK = np.zeros([len(RaAff), npoints])
    specHolderTh = np.zeros([len(RaAff), npoints])
    for i, k1Aff in enumerate(RaAff):
        for j, k5Aff in enumerate(affRange):
            SNKfun, SThfun = OPgenSpec(unkVecAff, scalesAff, k1Aff, k5Aff)
            specHolderNK[i, j] = SNKfun.eval()
            specHolderTh[i, j] = SThfun.eval()
        if i == 0:
            ax.plot(1 / affRange, specHolderNK[i, :], label="TReg/NK pSTAT5 w/ " + str(1 / RaAff[i]) + " IL2Ra Affinity", color="slateblue")
            ax.plot(1 / affRange, specHolderTh[i, :], label="TReg/Th pSTAT5 w/ " + str(1 / RaAff[i]) + " IL2Ra Affinity", color="orange")
        else:
            ax.plot(1 / affRange, specHolderNK[i, :], label="TReg/NK pSTAT5 w/ " + str(1 / RaAff[i]) + " IL2Ra Affinity", linestyle='dotted', color="slateblue")
            ax.plot(1 / affRange, specHolderTh[i, :], label="TReg/Th pSTAT5 w/ " + str(1 / RaAff[i]) + " IL2Ra Affinity", linestyle='dotted', color="orange")

    ax.set_xscale('log')
    ax.set_xlabel('Relative CD122/CD132 Affinity')
    ax.set_ylabel('Specificity')
    ax.legend()
