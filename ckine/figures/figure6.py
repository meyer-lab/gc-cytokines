"""
This creates Figure 6.
"""
import string
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import theano.tensor as T
import theano
from matplotlib.lines import Line2D
from .figureCommon import subplotLabel, getSetup, global_legend, calc_dose_response, import_pMuteins, catplot_comparison, nllsq_EC50, organize_expr_pred, mutein_scaling, plot_cells, plot_ligand_comp
from ..imports import import_pstat, import_samples_2_15, import_Rexpr
from ..model import getTotalActiveSpecies, receptor_expression, getRateVec, getparamsdict, getMutAffDict
from ..differencing_op import runCkineDoseOp
from ..tensor import perform_decomposition, find_R2X, z_score_values

unkVec_2_15, scales = import_samples_2_15(N=1)
data, receptor_data, cell_names_receptor = import_Rexpr()
ckineConc, cell_names_pstat, IL2_data, IL15_data, _ = import_pstat()
ckineC = ckineConc[7]
time = 240.
unkVec = getRateVec(unkVec_2_15)
CondIL = np.zeros((1, 6), dtype=np.float64)
CondIL[0, 0] = ckineC
Op = runCkineDoseOp(tt=np.array(time), condense=getTotalActiveSpecies().astype(np.float64), conditions=CondIL)
unkVec = unkVec[6::].flatten()
unkVecT = T.set_subtensor(T.zeros(54)[0:], np.transpose(unkVec))
mutData = import_pMuteins()


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7.5, 6), (3, 4), multz={1: 1})

    ax[2].axis('off')

    for ii, item in enumerate(ax):
        if ii < 2:
            subplotLabel(item, string.ascii_uppercase[ii])
        elif ii > 2:
            subplotLabel(item, string.ascii_uppercase[ii - 1])

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

    mutEC50df = get_Mut_EC50s()
    affComp(ax[0])
    calc_plot_specificity(ax[7], 'NK', df_spec, df_act, ckines, ckineConc_)
    calc_plot_specificity(ax[8], 'T-helper', df_spec, df_act, ckines, ckineConc_)
    global_legend(ax[8], Spec=True, Mut=True)
    Specificity(ax=ax[9])
    Spec_Aff(ax[10], 40, unkVecT, scalesT)
    catplot_comparison(ax[1], mutEC50df, legend=False, Mut=True)
    Mut_Fact(ax[3:7])
    legend = ax[3].get_legend()
    labels = (x.get_text() for x in legend.get_texts())
    ax[2].legend(legend.legendHandles, labels, loc='center left', prop={"size": 9})
    ax[3].get_legend().remove()

    return f


def affComp(ax):
    """Compare 2Ra and 2BGc dissociation constants of wild type and mutant IL-2s"""
    affdict = getMutAffDict()
    ligList = ['WT N-term', 'WT C-term', 'V91K C-term', 'R38Q N-term', 'F42Q N-Term', 'N88D C-term', 'WT IL2']
    RaAff, GcBAff = np.zeros([len(ligList)]), np.zeros([len(ligList)])

    for i in range(0, len(ligList) - 1):
        RaAff[i] = affdict[ligList[i]][0]
        GcBAff[i] = affdict[ligList[i]][1]

    RaAff[len(ligList) - 1] = unkVec[1] / 0.6
    GcBAff[len(ligList) - 1] = unkVec[4] / 0.6

    KDdf = pd.DataFrame({'RaAff': RaAff, 'GcBAff': GcBAff, 'Ligand': ligList})
    KDdf = KDdf.sort_values(by=["Ligand"])
    sns.scatterplot(x='RaAff', y='GcBAff', data=KDdf, hue="Ligand", palette=sns.color_palette("husl", 8)[0:5] + sns.color_palette("husl", 8)[6:8], ax=ax, legend=False)
    ax.set_xlabel("IL-2Rα KD (nM)")
    ax.set_ylabel("IL-2Rβ/γc KD (nM)")


def calc_plot_specificity(ax, cell_compare, df_specificity, df_activity, ligands, concs):
    """ Calculates and plots specificity for both cytokines and experimental/predicted activity for T-regs. """

    # calculate specificity and append to dataframe
    for _, ckine in enumerate(ligands):
        for _, conc in enumerate(concs):
            df_specificity = specificity(df_specificity, df_activity, 'T-reg', cell_compare, ckine, 60., conc)

    df_specificity["Concentration"] = np.log10(df_specificity["Concentration"].astype(np.float))  # logscale for plotting
    df_specificity["Specificity"] = np.log10(df_specificity["Specificity"].astype(np.float))
    df_specificity.drop(df_specificity[(df_specificity["Concentration"] < -2.5)].index, inplace=True)  # drop second conc due to negative activity

    # plot all specificty values
    sns.set_palette(sns.xkcd_palette(["violet", "goldenrod"]))
    sns.scatterplot(x="Concentration", y="Specificity", hue="Ligand", data=df_specificity.loc[(df_specificity["Cells"] ==
                                                                                               cell_compare) & (df_specificity["Data Type"] == 'Experimental')], ax=ax, marker='o', legend=False)
    sns.lineplot(x="Concentration", y="Specificity", hue="Ligand", data=df_specificity.loc[(df_specificity["Cells"] == cell_compare) &
                                                                                           (df_specificity["Data Type"] == 'Predicted')], ax=ax, legend=False)
    ax.set(xlabel="Concentration (nM)", ylabel="log$_{10}$[Specificity]", title=('T-reg vs. ' + cell_compare))


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

    colors = ["rich blue", "sun yellow"]
    sns.set_palette(sns.xkcd_palette(colors))

    sns.catplot(data=df, x='rate', y='value', kind="bar", hue='cell', ax=ax, legend=False)
    ax.set_ylabel("Value")
    ax.set_xlabel("")
    ax.legend()
    ax.set_yscale("symlog", linthreshy=0.01)
    labels = df.rate.unique()
    for ii, _ in enumerate(labels):
        labels[ii] = labels[ii].split(".")[-1]
    ax.set_xticklabels(labels, rotation=60, rotation_mode="anchor", ha="right", fontdict={'fontsize': 6})


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
    RaAff = np.array([1, 2])
    specHolderNK = np.zeros([len(RaAff), npoints])
    specHolderTh = np.zeros([len(RaAff), npoints])
    for i, k1Aff in enumerate(RaAff):
        for j, k5Aff in enumerate(affRange):
            SNKfun, SThfun = OPgenSpec(unkVecAff, scalesAff, k1Aff, k5Aff)
            specHolderNK[i, j] = SNKfun.eval()
            specHolderTh[i, j] = SThfun.eval()
        if i == 0:
            ax.plot(1 / affRange, specHolderNK[i, :], label="TReg/NK pSTAT5 w/ " + str(1 / RaAff[i]) + " IL2Ra Affinity", color="xkcd:rich blue")
            ax.plot(1 / affRange, specHolderTh[i, :], label="TReg/Th pSTAT5 w/ " + str(1 / RaAff[i]) + " IL2Ra Affinity", color="xkcd:sun yellow")
        else:
            ax.plot(1 / affRange, specHolderNK[i, :], label="TReg/NK pSTAT5 w/ " + str(1 / RaAff[i]) + " IL2Ra Affinity", linestyle='dotted', color="xkcd:rich blue")
            ax.plot(1 / affRange, specHolderTh[i, :], label="TReg/Th pSTAT5 w/ " + str(1 / RaAff[i]) + " IL2Ra Affinity", linestyle='dotted', color="xkcd:sun yellow")

    ax.set_xscale('log')
    ax.set_xlabel('Relative IL-2Rβ/γc Affinity')
    ax.set_ylabel('Specificity')
    handles = []
    line = Line2D([], [], color='black', marker='_', linestyle='None', markersize=6, label='WT IL2Ra Affinity')
    point = Line2D([], [], color='black', marker='.', linestyle='None', markersize=6, label='0.5 IL2Ra Affinity')
    handles.append(line)
    handles.append(point)
    ax.legend(handles=handles)


def get_Mut_EC50s():
    """Creates df with mutein EC50s included"""
    x0 = [1, 2., 1000.]
    concentrations = mutData.Concentration.unique()
    ligand_order = ['WT N-term', 'WT C-term', 'V91K C-term', 'R38Q N-term', 'F42Q N-Term', 'N88D C-term']
    celltypes = mutData.Cells.unique()
    times = mutData.Time.unique()
    EC50df = pd.DataFrame(columns=['Time Point', 'IL', 'Cell Type', 'Data Type', 'EC-50'])
    cell_groups = [['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+']]

    # experimental
    for _, IL in enumerate(ligand_order):
        for _, cell in enumerate(celltypes):
            for kk, timex in enumerate(times):
                doseData = np.array(mutData.loc[(mutData["Cells"] == cell) & (mutData["Ligand"] == IL) & (mutData["Time"] == timex)]["RFU"])
                EC50 = nllsq_EC50(x0, np.log10(concentrations.astype(np.float) * 10**4), doseData) - 4
                EC50df.loc[len(EC50df.index)] = pd.Series({'Time Point': timex, 'IL': IL, 'Cell Type': cell, 'Data Type': 'Experimental', 'EC-50': EC50})

    # predicted

    cell_order = ['NK', 'CD8+', 'T-reg', 'Naive Treg', 'Mem Treg', 'T-helper', 'Naive Th', 'Mem Th']

    df = pd.DataFrame(columns=['Cells', 'Ligand', 'Time Point', 'Concentration', 'Activity Type', 'Replicate', 'Activity'])

    # loop for each cell type and mutein
    for _, cell_name in enumerate(cell_order):

        IL2Ra = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == 'IL-2R$\\alpha$'), "Count"].values[0]
        IL2Rb = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == 'IL-2R$\\beta$'), "Count"].values[0]
        gc = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == '$\\gamma_{c}$'), "Count"].values[0]
        receptors = np.array([IL2Ra, IL2Rb, gc]).astype(np.float)

        for _, ligand_name in enumerate(ligand_order):
            # append dataframe with experimental and predicted activity
            df = organize_expr_pred(df, cell_name, ligand_name, receptors, ckineConc, tpsSc, unkVec_2_15)

    # determine scaling constants
    scalesIn = mutein_scaling(df, unkVec_2_15)

    # scale
    pred_data = np.zeros((12, 4, unkVec_2_15.shape[1]))
    for _, cell_name in enumerate(cell_order):
        for _, ligand_name in enumerate(ligand_order):
            for k, conc in enumerate(df.Concentration.unique()):
                for l, tp in enumerate(tpsSc):
                    pred_data[k, l] = df.loc[(df["Cells"] == cell_name) & (df["Ligand"] == ligand_name) & (
                        df["Activity Type"] == 'predicted') & (df["Concentration"] == conc) & (df["Time Point"] == tp) & (df["Replicate"] == 1)]["Activity"]

            for n, cell_names in enumerate(cell_groups):
                if cell_name in cell_names:
                    pred_data[:, :] = scalesIn[n, 1, 0] * pred_data[:, :] / (pred_data[:, :] + scalesIn[n, 0, 0])

            for kk, timeEC in enumerate(tpsSc):
                doseData = (pred_data[:, kk]).flatten()
                EC50 = nllsq_EC50(x0, np.log10(concentrations.astype(np.float) * 10**4), doseData) - 4
                EC50df.loc[len(EC50df.index)] = pd.Series({'Time Point': timeEC, 'IL': ligand_name, 'Cell Type': cell_name, 'Data Type': 'Predicted', 'EC-50': EC50})

    return EC50df


def Mut_Fact(ax):
    """Plots Non-Negative CP Factorization of Muteins into 4 ax subplots"""
    mutDataF = mutData.sort_values(by=['Cells', 'Ligand', 'Time', 'Concentration'])
    mutDataF.reset_index(inplace=True)
    _, _, _, _, WTdf = import_pstat()
    combdf = pd.concat([mutDataF, WTdf], sort=True, ignore_index=True)
    combdf = combdf.replace(["IL2", "IL15"], ["WT IL2", "WT IL15"])
    combdf = combdf.sort_values(by=['Cells', 'Ligand', 'Time', 'Concentration'])
    mutFactdf = combdf[(combdf.Cells != "Naive CD8+") & (combdf.Cells != "Mem CD8+")]
    mutTensor = np.reshape(mutFactdf["RFU"].values, (8, 8, 4, 12))  # cells, muteins/WT, times, and concs.

    concs = mutFactdf['Concentration'].unique()
    ts = mutFactdf['Time'].unique() / 60
    cells = mutFactdf['Cells'].unique()
    ligs = mutFactdf['Ligand'].unique()

    dataTensor = z_score_values(mutTensor, 0)
    parafac = perform_decomposition(dataTensor, 2, weightFactor=3)
    logging.info(find_R2X(dataTensor, parafac))

    # Ligands
    plot_ligand_comp(ax[0], parafac[1], 1, 2, ligs)
    ax[0].set_title("Ligands")
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlim(left=0)

    # Cells
    plot_cells(ax[1], parafac[0], 1, 2, cells)
    ax[1].legend()

    # Timepoints
    tComp = np.r_[np.zeros((1, 2)), parafac[2]]
    ts = np.append(np.array(0.0), ts)
    ax[2].set_xlabel("Time (hrs)")
    ax[2].set_ylabel("Component")
    ax[2].plot(ts, tComp)
    ax[2].set_xticks(np.array([0, 1, 2, 4]))
    ax[2].legend(["Component 1", "Component 2"])

    # Concentration
    ax[3].semilogx(concs, parafac[3])
    ax[3].set_xlabel("Concentration (nM)")
    ax[3].set_ylabel("Component")
    ax[3].set_xticks(np.array([10e-4, 10e-2, 10e0, 10e2]))
