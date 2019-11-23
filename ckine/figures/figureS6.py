"""
This creates Figure S6. Full Partial Derivative Bar
"""
import string
import numpy as np
import pandas as pd
import seaborn as sns
import theano.tensor as T
import theano
from .figureCommon import subplotLabel, getSetup, grouped_scaling
from ..model import getTotalActiveSpecies, receptor_expression, getRateVec, getparamsdict
from ..imports import import_Rexpr, import_samples_2_15, import_pstat
from ..differencing_op import runCkineDoseOp


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((30, 8), (2, 1))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    Specificity(ax=ax[0])
    Spec_Aff(ax[1], 40, unkVecT, scalesT)

    return f


unkVec, scales = import_samples_2_15(N=1)
ckineConc, _, IL2_data, IL15_data, _ = import_pstat()
_, receptor_data, cell_names_receptor = import_Rexpr()
ckineC = ckineConc[7]
time = 240.
unkVec = getRateVec(unkVec)
CondIL = np.zeros((1, 6), dtype=np.float64)
CondIL[0, 0] = ckineC
Op = runCkineDoseOp(tt=np.array(time), condense=getTotalActiveSpecies().astype(np.float64), conditions=CondIL)
unkVec = unkVec[6::].flatten()
unkVecT = T.set_subtensor(T.zeros(54)[0:], np.transpose(unkVec))


def genscalesT(unkVecOP):
    """ This generates the group of scaling constants for our OPs """
    cell_names_receptorC = cell_names_receptor.tolist()
    tps = np.array([0.5, 1., 2., 4.]) * 60
    Cond2 = np.zeros((1, 6), dtype=np.float64)
    Cond15 = np.zeros((1, 6), dtype=np.float64)
    pred2Vec, pred15Vec = np.zeros([len(cell_names_receptorC), ckineConc.size, 1, len(tps)]), np.zeros([len(cell_names_receptorC), ckineConc.size, 1, len(tps)])

    for i, Ctype in enumerate(cell_names_receptorC):  # Update each vec for unique cell expression levels
        cell_data = receptor_data[cell_names_receptorC.index(Ctype), :]
        unkVecOP = T.set_subtensor(unkVecOP[46], receptor_expression(cell_data[0], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # RA
        unkVecOP = T.set_subtensor(unkVecOP[47], receptor_expression(cell_data[1], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Rb
        unkVecOP = T.set_subtensor(unkVecOP[48], receptor_expression(cell_data[2], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Gc
        unkVecOP = T.set_subtensor(unkVecOP[49], 0)  # 15

        for j, conc in enumerate(ckineConc):
            Cond2[0, 0] = conc
            Cond15[0, 1] = conc
            for k, timeT in enumerate(tps):
                # calculate full tensor of predictions for all cells for all timepoints
                ScaleOp = runCkineDoseOp(tt=np.array(timeT), condense=getTotalActiveSpecies().astype(np.float64), conditions=Cond2)
                pred2Vec[i, j, 0, k] = ScaleOp(unkVecOP).eval()
                ScaleOp = runCkineDoseOp(tt=np.array(timeT), condense=getTotalActiveSpecies().astype(np.float64), conditions=Cond15)
                pred15Vec[i, j, 0, k] = ScaleOp(unkVecOP).eval()

    scalesTh = grouped_scaling(scales, cell_names_receptorC, IL2_data, IL15_data, pred2Vec, pred15Vec)

    return scalesTh


scalesT = genscalesT(unkVecT)


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

    sns.barplot(data=df, x='rate', y='value', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, rotation_mode="anchor", ha="right")
    ax.set_ylim(-0.1, 0.1)


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
