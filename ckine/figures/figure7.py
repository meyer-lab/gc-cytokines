"""
This creates Figure 7. Just internal.
"""
import string
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import theano.tensor as T
import theano
import matplotlib
from matplotlib.lines import Line2D
from .figureCommon import subplotLabel, getSetup, global_legend, calc_dose_response, import_pMuteins, nllsq_EC50, organize_expr_pred, mutein_scaling, plot_cells, plot_ligand_comp, Par_Plot_comparison
from .figure4 import WT_EC50s
from ..imports import import_pstat, import_samples_2_15, import_Rexpr
from ..model import getTotalActiveSpecies, receptor_expression, getRateVec, getparamsdict, getMutAffDict
from ..differencing_op import runCkineDoseOp
from ..tensor import perform_decomposition, find_R2X, z_score_values

unkVec_2_15, scales = import_samples_2_15(N=1)
data, receptor_data, cell_names_receptor = import_Rexpr()
ckineConc, cell_names_pstat, IL2_data, IL15_data, _ = import_pstat()
tpsSc = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
_, _, scalesT = calc_dose_response(cell_names_receptor, unkVec_2_15, scales, receptor_data, tpsSc, ckineConc, IL2_data, IL15_data)


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 5), (2, 3))
    _, receptor_dataC, cell_names_receptorC = import_Rexpr()


    #for ii, item in enumerate(ax):
    #        subplotLabel(item, string.ascii_uppercase[ii])

    Spec_Aff(ax[0:6], scalesT)

    return f


def OPgen(unkVecOP, CellTypes, OpC, scalesTh, RaAffM, RbAffM, CD25Add=1.0):
    "Generates the UnkVec with cell specific receptor abundances and expression rates"
    _, receptor_dataC, cell_names_receptorC = import_Rexpr()
    cell_names_receptorC = cell_names_receptorC.tolist()

    # set true Op
    cell_data = receptor_dataC[cell_names_receptorC.index(CellTypes), :]
    unkVecOP = T.set_subtensor(unkVecOP[46], receptor_expression(cell_data[0] * CD25Add, unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # RA
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


def OPgenSpec(unk, scalesIn, Op, k1Aff=1.0, k5Aff=1.0):
    """ Make an Op for specificity from the given conditions. """
    S_CD = (OPgen(unk, "T-reg", Op, scalesIn, k1Aff, k5Aff) /
            OPgen(unk, "CD8+", Op, scalesIn, k1Aff, k5Aff))

    S_ThAve = (OPgen(unk, "T-reg", Op, scalesIn, k1Aff, k5Aff) /
            OPgen(unk, "T-helper", Op, scalesIn, k1Aff, k5Aff, 1.0))

    S_ThHigh = (OPgen(unk, "T-reg", Op, scalesIn, k1Aff, k5Aff) /
            OPgen(unk, "T-helper", Op, scalesIn, k1Aff, k5Aff, 2.0))
    
    S_ThLow = (OPgen(unk, "T-reg", Op, scalesIn, k1Aff, k5Aff) /
            OPgen(unk, "T-helper", Op, scalesIn, k1Aff, k5Aff, 0.5))
    
    S_NK = (OPgen(unk, "T-reg", Op, scalesIn, k1Aff, k5Aff) /
            OPgen(unk, "NK", Op, scalesIn, k1Aff, k5Aff))

    return S_CD, S_ThAve, S_ThHigh, S_ThLow, S_NK


def Spec_Aff(ax, scalesAff):
    "Plots specificity for a cell type over a range of IL2RBG and IL2Ra affinities"
    dose = ckineConc[7]
    affRange = np.logspace(2, 0, 50)
    timemat = [2 * 60., 12 * 60.]
    specHolderCD = np.zeros([2, len(affRange)])
    specHolderThAve = np.zeros([2, len(affRange)])
    specHolderThHigh = np.zeros([2, len(affRange)])
    specHolderThLow = np.zeros([2, len(affRange)])
    specHolderNK = np.zeros([2, len(affRange)])
    CondIL = np.zeros((1, 6), dtype=np.float64)
    CondIL[0, 0] = dose

    unkVec = getRateVec(unkVec_2_15)
    unkVec = unkVec[6::].flatten()
    unkVecAff = T.set_subtensor(T.zeros(54)[0:], np.transpose(unkVec))
    k1affmat = np.array([1., 2.])
    for CD25aff in k1affmat:
        if CD25aff == 1.:
            linemarker = "solid"
        else:
            linemarker = 'dotted'

        for ii, time in enumerate(timemat):
            for jj, affinity in enumerate(affRange):
                Op = runCkineDoseOp(tt=np.array(time), condense=getTotalActiveSpecies().astype(np.float64), conditions=CondIL)
                SCD8fun, SThfunAve, SThfunHigh, SThfunLow, SNKfun = OPgenSpec(unkVecAff, scalesAff, Op, CD25aff, affinity)
                specHolderCD[ii, jj] = SCD8fun.eval()
                specHolderThAve[ii, jj] = SThfunAve.eval()
                specHolderThHigh[ii, jj] = SThfunHigh.eval()
                specHolderThLow[ii, jj] = SThfunLow.eval()
                specHolderNK[ii, jj] = SNKfun.eval()
                

    # 2 hours 
        ax[0].plot(1 / affRange, specHolderThLow[0, :], label="Low CD25 T-helpers", color="limegreen", linestyle=linemarker)
        ax[0].plot(1 / affRange, specHolderThAve[0, :], label="Average CD25 T-helpers", color="xkcd:rich blue", linestyle=linemarker)
        ax[0].plot(1 / affRange, specHolderThHigh[0, :], label="High CD25 T-helpers", color="xkcd:sun yellow", linestyle=linemarker)

        num = 0
        ax[num].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[num].legend(prop={'size': 7}, loc = 'upper left')
        ax[num].set_xlim(left = 10e-3, right = 10e-1)
        ax[num].set_xscale("log")
        ax[num].set_xlabel('Relative IL-2Rβ/γc Affinity')
        ax[num].set_ylabel("Activation Specificity")
        ax[num].set_ylim((0, 70))
        #ax[num].set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
        ax[num].set_title("T-reg/T-helper pSTAT5 - 2 Hours")


        ax[1].plot(1 / affRange, specHolderCD[0, :], color="darkorange", linestyle=linemarker)
        num = 1
        ax[num].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[num].set_xlim(left = 10e-3, right = 10e-1)
        ax[num].set_ylim((0, 250))
        ax[num].set_xscale("log")
        ax[num].set_xlabel('Relative IL-2Rβ/γc Affinity')
        ax[num].set_ylabel("Activation Specificity")
        #ax[num].set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
        ax[num].set_title("T-reg/CD8+ pSTAT5 - 2 Hours")

        ax[2].plot(1 / affRange, specHolderNK[0, :], color="orangered", linestyle=linemarker)
        num = 2
        ax[num].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[num].set_xlim(left = 10e-3, right = 10e-1)
        ax[num].set_ylim((0, 250))
        ax[num].set_xscale("log")
        ax[num].set_xlabel('Relative IL-2Rβ/γc Affinity')
        ax[num].set_ylabel("Activation Specificity")
        #ax[num].set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
        ax[num].set_title("T-reg/NK pSTAT5 - 2 Hours")

    #12 hours
        ax[3].plot(1 / affRange, specHolderThLow[0, :], label="Low CD25 T-helpers", color="limegreen", linestyle=linemarker)
        ax[3].plot(1 / affRange, specHolderThAve[1, :], label="Average CD25 T-helpers", color="xkcd:rich blue", linestyle=linemarker)
        ax[3].plot(1 / affRange, specHolderThHigh[1, :], label="High CD25 T-helpers", color="xkcd:sun yellow", linestyle=linemarker)

        num = 3
        ax[num].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[num].legend(prop={'size': 7}, loc = 'upper left')
        ax[num].set_xlim(left = 10e-3, right = 10e-1)
        ax[num].set_xscale("log")
        ax[num].set_xlabel('Relative IL-2Rβ/γc Affinity')
        ax[num].set_ylabel("Activation Specificity")
        ax[num].set_ylim((0, 70))
        #ax[num].set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
        ax[num].set_title("T-reg/T-helper pSTAT5 - 12 Hours")

        ax[4].plot(1 / affRange, specHolderCD[1, :], color="darkorange", linestyle=linemarker)
        num = 4
        ax[num].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[num].set_xlim(left = 10e-3, right = 10e-1)
        ax[num].set_xscale("log")
        ax[num].set_ylim((0, 250))
        ax[num].set_xlabel('Relative IL-2Rβ/γc Affinity')
        ax[num].set_ylabel("Activation Specificity")
        #ax[num].set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
        ax[num].set_title("T-reg/CD8+ pSTAT5 - 12 Hours")


        ax[5].plot(1 / affRange, specHolderNK[1, :], color="orangered", linestyle=linemarker)
        num = 5
        ax[num].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[num].set_xlim(left = 10e-3, right = 10e-1)
        ax[num].set_ylim((0, 250))
        ax[num].set_xscale("log")
        ax[num].set_xlabel('Relative IL-2Rβ/γc Affinity')
        ax[num].set_ylabel("Activation Specificity")
        #ax[num].set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
        ax[num].set_title("T-reg/NK pSTAT5 - 12 Hours")

    line = Line2D([], [], color='black', marker='_', linestyle='None', markersize=6, label='WT CD25 Affinity')
    point = Line2D([], [], color='black', marker='.', linestyle='None', markersize=6, label='0.5 CD25 Affinity')
    
    handles, _ = ax[0].get_legend_handles_labels()
    handles.append(line)
    handles.append(point)
    ax[0].legend(prop={'size': 7}, loc = 'upper left', handles=handles[-5::])
    
    ax[3].legend(prop={'size': 7}, loc = 'upper left', handles=handles[-5::])

    handles = []
    line = Line2D([], [], color='darkorange', marker='_', linestyle='None', markersize=6, label='WT CD25 Affinity')
    point = Line2D([], [], color='darkorange', marker='.', linestyle='None', markersize=6, label='0.5 CD25 Affinity')
    handles.append(line)
    handles.append(point)
    ax[1].legend(prop={'size': 7}, loc = 'upper left', handles=handles)
    ax[4].legend(prop={'size': 7}, loc = 'upper left', handles=handles)

    handles = []
    line = Line2D([], [], color='orangered', marker='_', linestyle='None', markersize=6, label='WT CD25 Affinity')
    point = Line2D([], [], color='orangered', marker='.', linestyle='None', markersize=6, label='0.5 CD25 Affinity')
    handles.append(line)
    handles.append(point)
    ax[2].legend(prop={'size': 7}, loc = 'upper right', handles=handles)
    ax[5].legend(prop={'size': 7}, loc = 'upper right', handles=handles)
