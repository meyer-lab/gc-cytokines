"""
This creates Figure S5. Full panel of measured vs simulated for IL-2 and IL-15.
"""
import string
import numpy as np
import matplotlib.cm as cm
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .figureCommon import subplotLabel, getSetup, plot_conf_int, plot_scaled_pstat
from ..model import runCkineUP, getTotalActiveSpecies, receptor_expression, nParams
from ..imports import import_Rexpr, import_pstat, import_samples_2_15
from ..differencing_op import runCkineDoseOp


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((20, 4), (1, 1))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    Specificity(ax=ax[0])

    return f


def Specificity(ax, cell_subset=None):
    """ Creates Theano Function for calculating Specificity gradient with respect to various parameters"""
    S_partials = Sfunc(unkVec.flatten()) / S.eval()
    y_pos = np.arange(nParams() - 6)
    vars_string = ['kfwd', 'krev4', 'krev5', 'krev16', 'krev17', 'krev22', 'krev23', 'krev27', 'krev31', 'krev33', 'krev35', 'endo',
                   'Aendo', 'sortF', 'kRec', 'kDeg', 'Rexpr2Ra', 'Rexpr2Rb', 'RexprGC', 'Rexpr15Ra', 'Rexpr7Ra', 'Rexpr9R', 'Rexpr4Ra', 'Rexp21Ra']
    Derivs = ax.bar(y_pos, S_partials, width=1, align='center', alpha=1)
    ax.set_ylim(-30, 30)
    barlabel(Derivs, vars_string, ax)


def OPgen(unkVec, CellTypes, Op):
    _, receptor_data, cell_names_receptor = import_Rexpr()
    cell_names_receptor = cell_names_receptor.tolist()
    OpHolder = np.zeros(len(CellTypes))
    CellTypes = [CellTypes]
    predVec = T.zeros(1)

    for i, Ctype in enumerate(CellTypes):  # Update each vec for unique cell expression levels
        cell_data = receptor_data[cell_names_receptor.index(Ctype), :]
        unkVec = T.set_subtensor(unkVec[16], receptor_expression(cell_data[0], unkVec[11], unkVec[14], unkVec[13], unkVec[15]))
        unkVec = T.set_subtensor(unkVec[17], receptor_expression(cell_data[1], unkVec[11], unkVec[14], unkVec[13], unkVec[15]))
        unkVec = T.set_subtensor(unkVec[18], receptor_expression(cell_data[2], unkVec[11], unkVec[14], unkVec[13], unkVec[15]))
        unkVec = T.set_subtensor(unkVec[19], 0)
        Cell_Op = Op(unkVec)

    return Cell_Op


def barlabel(rects, labels, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if height < 0:
            height = 0
        ax.annotate(labels[i],
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


unkVec, _ = import_samples_2_15(N=1)
CondIL = np.zeros((1, 6), dtype=np.float64)
CondIL[0] = 1.
Op = runCkineDoseOp(tt=np.array(500.), condense=getTotalActiveSpecies().astype(np.float64), conditions=CondIL)
unkVecTrunc = T.zeros(24)
unkVec = unkVec[6::].flatten()
unkVecTrunc = T.set_subtensor(unkVecTrunc[0:], np.transpose(unkVec))
unkVecT = unkVecTrunc
_, receptor_data, cell_names_receptor = import_Rexpr()
S = (OPgen(unkVecT, "T-reg", Op) /
     (OPgen(unkVecT, "T-reg", Op) +
      OPgen(unkVecT, "Naive Treg", Op) +
      OPgen(unkVecT, "Mem Treg", Op) +
      OPgen(unkVecT, "T-helper", Op) +
      OPgen(unkVecT, "Naive Th", Op) +
      OPgen(unkVecT, "Mem Th", Op) +
      OPgen(unkVecT, "NK", Op) +
      OPgen(unkVecT, "CD8+", Op) +
      OPgen(unkVecT, "Naive CD8+", Op) +
      OPgen(unkVecT, "Mem CD8+", Op)))
Sgrad = T.grad(S[0], unkVecT)
Sfunc = theano.function([unkVecT], Sgrad)
