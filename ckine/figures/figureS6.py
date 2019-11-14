"""
This creates Figure S6. Full Partial Derivative Bar
"""
import string
import numpy as np
import theano.tensor as T
import theano
from .figureCommon import subplotLabel, getSetup
from ..model import getTotalActiveSpecies, receptor_expression, getRateVec, getparamsdict, rxParams
from ..imports import import_Rexpr, import_samples_2_15
from ..differencing_op import runCkineDoseOp


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((30, 4), (1, 1))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    Specificity(ax=ax[0])

    return f


def Specificity(ax):
    """ Creates Theano Function for calculating Specificity gradient with respect to various parameters"""
    S_partials = Sfunc(unkVec.flatten()) / S.eval()
    print(S.eval())
    y_pos = np.arange(rxParams() - 6)
    vars_string = ['kfwd', 'krev4', 'krev5', 'krev16', 'krev17', 'krev22', 'krev23', 'krev27', 'krev31', 'krev33', 'krev35', 'endo',
                   'Aendo', 'sortF', 'kRec', 'kDeg', 'Rexpr2Ra', 'Rexpr2Rb', 'RexprGC', 'Rexpr15Ra', 'Rexpr7Ra', 'Rexpr9R', 'Rexpr4Ra', 'Rexp21Ra']
    ax.set_ylim(-0.1, 0.1)
    Derivs = ax.bar(y_pos, S_partials, width=1, align='center', alpha=1)
    barlabel(Derivs, vars_string, ax)


def OPgen(unkVecOP, CellTypes, OpC):
    "Generates the UnkVec with cell specific receptor abundances and expression rates"
    _, receptor_dataC, cell_names_receptorC = import_Rexpr()
    cell_names_receptorC = cell_names_receptorC.tolist()
    CellTypes = [CellTypes]

    for Ctype in CellTypes:  # Update each vec for unique cell expression levels
        cell_data = receptor_dataC[cell_names_receptorC.index(Ctype), :]
        print(CellTypes)
        print(receptor_dataC[cell_names_receptorC.index(Ctype), :])
        unkVecOP = T.set_subtensor(unkVecOP[46], receptor_expression(cell_data[0], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # RA
        unkVecOP = T.set_subtensor(unkVecOP[47], receptor_expression(cell_data[1], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Rb
        unkVecOP = T.set_subtensor(unkVecOP[48], receptor_expression(cell_data[2], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Gc
        unkVecOP = T.set_subtensor(unkVecOP[49], 0)  # 15
        print()
        #unkVecOP = T.set_subtensor(unkVecOP[4], unkVecOP[4]*100)  # 15
        #unkVecOP = T.set_subtensor(unkVecOP[0], unkVecOP[0]*100)  # 15
        Cell_Op = OpC(unkVecOP)

    return Cell_Op


def barlabel(rects, labels, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    ODunk = np.zeros(60)
    OD = getparamsdict(ODunk)
    labels = list(OD.keys())
    labels = labels[6::]
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if height < 0 or height > 0.1:
            height = 0
        ax.annotate(labels[i],
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(-5, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='left', va='bottom',
                    rotation=45)


unkVec, _ = import_samples_2_15(N=1)
unkVec = getRateVec(unkVec)
CondIL = np.zeros((1, 6), dtype=np.float64)
CondIL[0, 0] = 2.59588659e-01
Op = runCkineDoseOp(tt=np.array(240.), condense=getTotalActiveSpecies().astype(np.float64), conditions=CondIL)
unkVecTrunc = T.zeros(54)
unkVec = unkVec[6::].flatten()
unkVecTrunc = T.set_subtensor(unkVecTrunc[0:], np.transpose(unkVec))
unkVecT = unkVecTrunc
_, receptor_data, cell_names_receptor = import_Rexpr()
print(OPgen(unkVecT, "T-reg", Op).eval())
print(OPgen(unkVecT, "Naive Treg", Op).eval())
print(OPgen(unkVecT, "Mem Treg", Op).eval())
print(OPgen(unkVecT, "T-helper", Op).eval())
print(OPgen(unkVecT, "Naive Th", Op).eval())
print(OPgen(unkVecT, "Mem Th", Op).eval())
print(OPgen(unkVecT, "NK", Op).eval())
print(OPgen(unkVecT, "CD8+", Op).eval())
print(OPgen(unkVecT, "Naive CD8+", Op).eval())
print(OPgen(unkVecT, "Mem CD8+", Op).eval())


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
