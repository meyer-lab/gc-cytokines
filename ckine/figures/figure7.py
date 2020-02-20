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
    ax, f = getSetup((6, 5), (2, 2))
    _, receptor_dataC, cell_names_receptorC = import_Rexpr()


    for ii, item in enumerate(ax):
            subplotLabel(item, string.ascii_uppercase[ii])

    Spec_Aff(ax[0:4], scalesT)

    return f


def OPgen(unkVecOP, CellTypes, OpC, scalesTh, CD25Add=1.0):
    "Generates the UnkVec with cell specific receptor abundances and expression rates"
    _, receptor_dataC, cell_names_receptorC = import_Rexpr()
    cell_names_receptorC = cell_names_receptorC.tolist()

    # set true Op
    cell_data = receptor_dataC[cell_names_receptorC.index(CellTypes), :]
    unkVecOP = T.set_subtensor(unkVecOP[46], receptor_expression(cell_data[0] * CD25Add, unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # RA
    unkVecOP = T.set_subtensor(unkVecOP[47], receptor_expression(cell_data[1], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Rb
    unkVecOP = T.set_subtensor(unkVecOP[48], receptor_expression(cell_data[2], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Gc
    unkVecOP = T.set_subtensor(unkVecOP[49], 0)


    cell_groups = np.array([['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+', 'Naive CD8+', 'Mem CD8+']])
    for i, group in enumerate(cell_groups):
        group = np.array(group)
        if np.where(group == CellTypes)[0].size > 0:
            scale1 = scalesTh[i, 1, 0]
            scale2 = scalesTh[i, 0, 0]

    Cell_Op = (OpC(unkVecOP) * scale1) / (OpC(unkVecOP) + scale2)

    return Cell_Op


def OPgenSpec(unk, scalesIn, Op):
    """ Make an Op for specificity from the given conditions. """
    S_NK = (OPgen(unk, "T-reg", Op, scalesIn) /
            OPgen(unk, "CD8+", Op, scalesIn))

    S_ThAve = (OPgen(unk, "T-reg", Op, scalesIn) /
            OPgen(unk, "T-helper", Op, scalesIn, 1.0))

    S_ThHigh = (OPgen(unk, "T-reg", Op, scalesIn) /
            OPgen(unk, "T-helper", Op, scalesIn, 2.0))
    
    S_ThLow = (OPgen(unk, "T-reg", Op, scalesIn) /
            OPgen(unk, "T-helper", Op, scalesIn, 0.5))

    return S_NK, S_ThAve, S_ThHigh, S_ThLow


def Spec_Aff(ax, scalesAff):
    "Plots specificity for a cell type over a range of IL2RBG and IL2Ra affinities"
    dosemat = np.logspace(-4, 2, 24)
    timemat = [2 * 60., 12 * 60.]
    specHolderCD = np.zeros([2, len(dosemat)])
    specHolderThAve = np.zeros([2, len(dosemat)])
    specHolderThHigh = np.zeros([2, len(dosemat)])
    specHolderThLow = np.zeros([2, len(dosemat)])

    unkVec = getRateVec(unkVec_2_15)
    unkVec = unkVec[6::].flatten()
    unkVecAff = T.set_subtensor(T.zeros(54)[0:], np.transpose(unkVec))

    for ii, time in enumerate(timemat):

        for jj, dose in enumerate(dosemat):
            CondIL = np.zeros((1, 6), dtype=np.float64)
            CondIL[0, 0] = dose
            Op = runCkineDoseOp(tt=np.array(time), condense=getTotalActiveSpecies().astype(np.float64), conditions=CondIL)

            SCD8fun, SThfunAve, SThfunHigh, SThfunLow = OPgenSpec(unkVecAff, scalesAff, Op)
            specHolderCD[ii, jj] = SCD8fun.eval()
            specHolderThAve[ii, jj] = SThfunAve.eval()
            specHolderThHigh[ii, jj] = SThfunHigh.eval()
            specHolderThLow[ii, jj] = SThfunLow.eval()
            
# 2 hours 
    ax[0].plot(dosemat, specHolderThLow[0, :], label="Low CD25 T-helpers", color="limegreen")
    ax[0].plot(dosemat, specHolderThAve[0, :], label="Average CD25 T-helpers", color="xkcd:rich blue")
    ax[0].plot(dosemat, specHolderThHigh[0, :], label="High CD25 T-helpers", color="xkcd:sun yellow")

    num = 0
    ax[num].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[num].legend()
    ax[num].set_xlim(left = 10e-5, right = 10e1)
    ax[num].set_xscale("log")
    ax[num].set_xlabel("IL-2 Concentration (nM)")
    ax[num].set_ylabel("Activation Specificity")
    ax[num].set_ylim((0, 50))
    ax[num].set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
    ax[num].set_title("TReg/T-helper pSTAT5 - 2 Hours")

    
    ax[2].plot(dosemat, specHolderCD[0, :], label="TReg/CD8+ pSTAT5", color="darkorange")
    num = 2
    ax[num].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[num].set_xlim(left = 10e-5, right = 10e1)
    ax[num].set_ylim((0, 150))
    ax[num].set_xscale("log")
    ax[num].set_xlabel("IL-2 Concentration (nM)")
    ax[num].set_ylabel("Activation Specificity")
    ax[num].set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
    ax[num].set_title("TReg/CD8+ pSTAT5 - 2 Hours")

#12 hours
    ax[1].plot(dosemat, specHolderThLow[0, :], label="Low CD25 T-helpers", color="limegreen")
    ax[1].plot(dosemat, specHolderThAve[1, :], label="Average CD25 T-helpers", color="xkcd:rich blue")
    ax[1].plot(dosemat, specHolderThHigh[1, :], label="High CD25 T-helpers", color="xkcd:sun yellow")

    num = 1
    ax[num].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[num].legend()
    ax[num].set_xlim(left = 10e-5, right = 10e1)
    ax[num].set_xscale("log")
    ax[num].set_xlabel("IL-2 Concentration (nM)")
    ax[num].set_ylabel("Activation Specificity")
    ax[num].set_ylim((0, 50))
    ax[num].set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
    ax[num].set_title("TReg/T-helper pSTAT5 - 12 Hours")

    ax[3].plot(dosemat, specHolderCD[1, :], color="darkorange")
    num = 3
    ax[num].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[num].set_xlim(left = 10e-5, right = 10e1)
    ax[num].set_xscale("log")
    ax[num].set_ylim((0, 150))
    ax[num].set_xlabel("IL-2 Concentration (nM)")
    ax[num].set_ylabel("Activation Specificity")
    ax[num].set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
    ax[num].set_title("TReg/CD8+ pSTAT5 - 12 Hours")
    
