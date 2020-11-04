"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_uppercase
from os.path import join, dirname
import seaborn as sns
import numpy as np
import pandas as pds
import matplotlib
import matplotlib.cm as cm
import svgutils.transform as st
from scipy import stats
from matplotlib import gridspec, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import least_squares
from ..tensor import find_R2X
from ..imports import import_pstat
from ..model import runCkineUP, getTotalActiveSpecies, receptor_expression


path_here = dirname(dirname(__file__))


matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 5
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35


def getSetup(figsize, gridd, multz=None, empts=None):
    """ Establish figure set-up with subplots. """
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x: x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def set_bounds(ax):
    """ Set bounds of component plots. """
    x_max = np.max(np.absolute(np.asarray(ax.get_xlim()))) * 1.1
    y_max = np.max(np.absolute(np.asarray(ax.get_ylim()))) * 1.1

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)


def plot_R2X(ax, tensor, factors_list):
    """Function to plot R2X bar graph."""
    R2X_array = list()
    for _, factors in enumerate(factors_list):
        R2X_array.append(find_R2X(tensor, factors))

    ax.plot(range(1, len(factors_list) + 1), R2X_array, "ko", label="Overall R2X")
    ax.set_ylabel("R2X")
    ax.set_xlabel("Number of Components")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(1, len(factors_list) + 1))
    ax.set_xticklabels(np.arange(1, len(factors_list) + 1))


def subplotLabel(axs, hstretch=None):
    """ Place subplot labels on figure. """
    if hstretch is None:
        hstretch = {}

    for ii, ax in enumerate(axs):
        hh = hstretch[ii] if ii in hstretch.keys() else 1.0

        if ii < 26:
            letter = ascii_uppercase[ii]
        else:
            letter = "A" + ascii_uppercase[ii - 26]

        ax.text(-0.2 / hh, 1.2, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")


def traf_names():
    """ Returns a list of the trafficking parameters in order they appear within unkVec. """
    return ["Endocyt. Rate Inact.", "Endocyt. Rate Act.", "Recycling Rate", "Degradation Rate"]


def plot_conf_int(ax, x_axis, y_axis, color, label=None):
    """ Shades the 25-75 percentiles dark and the 10-90 percentiles light. The percentiles are found along axis=1. """
    y_axis_top = np.percentile(y_axis, 90.0, axis=1)
    y_axis_bot = np.percentile(y_axis, 10.0, axis=1)
    ax.fill_between(x_axis, y_axis_top, y_axis_bot, color=color, alpha=0.2, linewidth = 0)

    y_axis_top = np.percentile(y_axis, 75.0, axis=1)
    y_axis_bot = np.percentile(y_axis, 25.0, axis=1)
    ax.fill_between(x_axis, y_axis_top, y_axis_bot, color=color, alpha=0.65, label=label, linewidth = 0)
    if label is not None:
        ax.legend()


def plot_cells(ax, factors, component_x, component_y, cell_names, legend=True):
    """This function plots the combination decomposition based on cell type."""
    colors = cm.rainbow(np.linspace(0, 1, 10))
    if len(cell_names) == 10:
        markersCells = ["^", "*", "D", "s", "X", "o", "4", "H", "P", "*", "D", "s", "X"]
    else:
        markersCells = ["*", "*", "4", "^", "P", "o", "H", "X"]
        colors = [colors[1], colors[9], colors[6], colors[0], colors[8], colors[5], colors[7], colors[4]]

    for ii, _ in enumerate(factors[:, component_x - 1]):
        ax.scatter(factors[ii, component_x - 1], factors[ii, component_y - 1], c=[colors[ii]], marker=markersCells[ii], label=cell_names[ii])

    ax.set_title("Cells")
    ax.set_xlabel("Component " + str(component_x))
    ax.set_ylabel("Component " + str(component_y))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if legend:
        ax.legend(prop={"size": 6})


def plot_ligand_comp(ax, factors, component_x, component_y, ligand_names):
    """This function plots the combination decomposition based on ligand type."""
    Comp1, Comp2 = np.zeros([len(ligand_names)]), np.zeros([len(ligand_names)])
    for ii, _ in enumerate(factors[:, component_x - 1]):
        Comp1[ii] = factors[ii, component_x - 1]
        Comp2[ii] = factors[ii, component_y - 1]

    CompDF = pds.DataFrame({"Comp1": Comp1, "Comp2": Comp2, "Ligand": ligand_names})
    sns.scatterplot(x="Comp1", y="Comp2", data=CompDF, hue="Ligand", palette=sns.color_palette("husl", 6), legend="full", ax=ax)
    ax.set_title("Ligands")
    ax.set_xlabel("Component " + str(component_x))
    ax.set_ylabel("Component " + str(component_y))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1, scale_x=1, scale_y=1):
    """ Add cartoon to a figure file. """

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale=scalee)
    cartoon.scale_xy(scale_x, scale_y)

    template.append(cartoon)
    template.save(figFile)


def plot_ligands(ax, factors, ligand_names, cutoff=0.0, compLabel=True):
    """Function to put all ligand decomposition plots in one figure."""
    ILs, _, _, _, _ = import_pstat()  # Cytokine stimulation concentrations in nM
    n_ligands = len(ligand_names)
    ILs = np.flip(ILs)
    colors = sns.color_palette()
    legend_shape = []
    markers = [".", "^", "d", "*"]

    for ii, name in enumerate(ligand_names):
        legend_shape.append(Line2D([0], [0], color="k", marker=markers[ii], label=name, linestyle=""))  # Make ligand legend elements

    for ii in range(factors.shape[1]):
        componentLabel = bool(compLabel)

        for jj in range(n_ligands):
            idx = range(jj * len(ILs), (jj + 1) * len(ILs))

            # If the component value never gets over cutoff, then don't plot the line
            if np.linalg.norm(factors[idx, ii]) > cutoff:
                if componentLabel:
                    ax.plot(ILs, factors[idx, ii], color=colors[ii], label="Cmp. " + str(ii + 1))
                    componentLabel = False
                else:
                    ax.plot(ILs, factors[idx, ii], color=colors[ii])
                ax.scatter(ILs, factors[idx, ii], color=colors[ii], marker=markers[jj])

    ax.add_artist(ax.legend(handles=legend_shape, loc=2))

    ax.set_xlabel("Ligand Concentration (nM)")
    ax.set_ylabel("Component")
    ax.set_xscale("log")
    ax.set_title("Ligands")
    ax.set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
    ax.set_ylim(0, 100)

    # Place legend
    ax.legend(loc=6)


def plot_timepoints(ax, ts, factors):
    """Function to put all timepoint plots in one figure."""
    ts = ts / 60
    colors = sns.color_palette()
    for ii in range(factors.shape[1]):
        ax.plot(ts, factors[:, ii], c=colors[ii], label="Component " + str(ii + 1))

    ax.set_xlabel("Time (hrs)")
    ax.set_ylabel("Component")
    ax.set_title("Time")
    ax.set_xticks(np.array([0, 1, 2, 4]))
    ax.legend()


def legend_2_15(ax, location="center right"):
    """ Plots a legend for all the IL-2 and IL-15 related plots in its own subpanel. """
    legend_elements = [
        Patch(facecolor="darkorchid", label="IL-2"),
        Patch(facecolor="goldenrod", label="IL-15"),
        Line2D([0], [0], marker="o", color="w", label="IL-2Rα+", markerfacecolor="k", markersize=16),
        Line2D([0], [0], marker="^", color="w", label="IL-2Rα-", markerfacecolor="k", markersize=16),
    ]
    ax.legend(handles=legend_elements, loc=location, prop={"size": 16})
    ax.axis("off")  # remove the grid


def plot_scaled_pstat(ax, cytokC, pstat, allFour=True):
    """ Plots pSTAT5 data scaled by the average activity measurement. """
    # plot pstat5 data for each time point
    colors = ["indigo", "teal", "forestgreen", "darkred"]
    for i in range(0, pstat.shape[0]):
        ax.scatter(cytokC, pstat[i, :], c=colors[i], s=2)  # 0.5 hr


def global_legend(ax, Spec=False, Mut=False, exppred=True):
    """ Create legend for colors and markers in subplots A-C. """
    purple = Patch(color="darkorchid", label="IL-2")
    yellow = Patch(color="goldenrod", label="IL-15")
    if not exppred:
        ax.legend(handles=[purple, yellow], loc="upper left")
    else:
        if not Mut:
            if not Spec:
                circle = Line2D([], [], color="black", marker="o", linestyle="None", markersize=6, label="Experimental")
                triangle = Line2D([], [], color="black", marker="^", linestyle="None", markersize=6, label="Predicted")
                ax.legend(handles=[purple, yellow, circle, triangle], bbox_to_anchor=(1.02, 1), loc="upper left")
            if Spec:
                circle = Line2D([], [], color="black", marker="o", linestyle="None", markersize=6, label="Experimental")
                line = Line2D([], [], color="black", marker="_", linestyle="None", markersize=6, label="Predicted")
                ax.legend(handles=[purple, yellow, circle, line], bbox_to_anchor=(1.02, 1), loc="upper left")
        if Mut:
            if not Spec:
                circle = Line2D([], [], color="black", marker="o", linestyle="None", markersize=6, label="Experimental")
                triangle = Line2D([], [], color="black", marker="^", linestyle="None", markersize=6, label="Predicted")
                ax.legend(handles=[purple, yellow, circle, triangle], loc="upper left")
            if Spec:
                circle = Line2D([], [], color="black", marker="o", linestyle="None", markersize=6, label="Experimental")
                line = Line2D([], [], color="black", marker="_", linestyle="None", markersize=6, label="Predicted")
                ax.legend(handles=[purple, yellow, circle, line], loc="upper left")


def calc_dose_response(cell_names, unkVec, receptor_data, tps, cytokC, expr_act2, expr_act15):
    """ Calculates activity for all cell types at various cytokine concentrations and timepoints. """
    PTS = cytokC.shape[0]  # number of cytokine concentrations

    rxntfr2 = unkVec.T.copy()
    total_activity2 = np.zeros((len(cell_names), PTS, rxntfr2.shape[0], tps.size))
    total_activity15 = total_activity2.copy()

    for i, cell in enumerate(cell_names):
        # updates rxntfr for receptor expression for IL2Ra, IL2Rb, gc
        cell_data = receptor_data[i]
        rxntfr2[:, 22] = receptor_expression(cell_data[0], rxntfr2[:, 17], rxntfr2[:, 20], rxntfr2[:, 19], rxntfr2[:, 21])
        rxntfr2[:, 23] = receptor_expression(cell_data[1], rxntfr2[:, 17], rxntfr2[:, 20], rxntfr2[:, 19], rxntfr2[:, 21])
        rxntfr2[:, 24] = receptor_expression(cell_data[2], rxntfr2[:, 17], rxntfr2[:, 20], rxntfr2[:, 19], rxntfr2[:, 21])
        rxntfr2[:, 25] = 0.0  # We never observed any IL-15Ra

        rxntfr15 = rxntfr2.copy()

        # loop for each IL2 concentration
        for j in range(PTS):
            rxntfr2[:, 0] = rxntfr15[:, 1] = cytokC[j]  # assign concs for each cytokine

            # handle case of IL-2
            yOut = runCkineUP(tps, rxntfr2)
            activity2 = np.dot(yOut, getTotalActiveSpecies().astype(np.float))
            # handle case of IL-15
            yOut = runCkineUP(tps, rxntfr15)
            activity15 = np.dot(yOut, getTotalActiveSpecies().astype(np.float))

            total_activity2[i, j, :, :] = np.reshape(activity2, (-1, len(tps)))  # save the activity from this concentration for all 4 tps
            total_activity15[i, j, :, :] = np.reshape(activity15, (-1, len(tps)))  # save the activity from this concentration for all 4 tps

    return total_activity2, total_activity15


def import_pMuteins():
    """ Loads CSV file containing pSTAT5 levels from Visterra data for muteins. """
    data = pds.read_csv(join(path_here, "data/Monomeric_mutein_pSTAT_data.csv"), encoding="latin1")
    data["RFU"] = data["RFU"] - data.groupby(["Cells", "Ligand"])["RFU"].transform("min")

    for conc in data.Concentration.unique():
        data = data.replace({"Concentration": conc}, np.round(conc, decimals=9))

    namedict = {
        "IL2-060 monomeric": "WT N-term",
        "Cterm IL-2 monomeric WT": "WT C-term",
        "Cterm IL-2 monomeric V91K": "V91K C-term",
        "IL2-109 monomeric": "R38Q N-term",
        "IL2-110 monomeric": "F42Q N-Term",
        "Cterm N88D monomeric": "N88D C-term",
    }
    data = data.replace({"Ligand": namedict})

    return data


dataMean = import_pMuteins()
dataMean.reset_index(inplace=True)
_, _, _, _, pstat_df = import_pstat()
dataMean = dataMean.append(pstat_df, ignore_index=True, sort=True)


def calc_dose_response_mutein(unkVec, tps, muteinC, mutein_name, cell_receptors):
    """ Calculates activity for a given cell type at various mutein concentrations and timepoints. """
    unkVec[22:25] = cell_receptors[0:3]

    unkVec = np.repeat(np.atleast_2d(unkVec).T, len(muteinC), axis=1).T
    unkVec[:, 0] = muteinC
    yOut = runCkineUP(tps, unkVec, mut_name=mutein_name)
    active_ckine = np.dot(yOut, getTotalActiveSpecies().astype(np.float))

    return active_ckine.reshape(len(muteinC), len(tps))


def organize_expr_pred(df, cell_name, ligand_name, receptors, muteinC, tps, unkVec):
    """ Appends input dataframe with experimental and predicted activity for a given cell type and mutein. """
    num = len(tps) * len(muteinC)

    # organize experimental pstat data
    exp_data = np.zeros((12, 4))
    mutein_conc = exp_data.copy()
    for i, conc in enumerate(dataMean.Concentration.unique()):
        exp_data[i, :] = np.array(dataMean.loc[(dataMean["Cells"] == cell_name) & (dataMean["Ligand"] == ligand_name) & (dataMean["Concentration"] == conc), "RFU"])
        mutein_conc[i, :] = conc

    df_exp = pds.DataFrame(
        {
            "Cells": np.tile(np.array(cell_name), num),
            "Ligand": np.tile(np.array(ligand_name), num),
            "Time Point": np.tile(tps, 12),
            "Concentration": mutein_conc.reshape(num),
            "Activity Type": np.tile(np.array("experimental"), num),
            "Replicate": np.zeros((num)),
            "Activity": exp_data.reshape(num),
        }
    )
    df = df.append(df_exp, ignore_index=True)

    # calculate predicted dose response
    pred_data = np.zeros((12, 4, unkVec.shape[1]))
    for j in range(unkVec.shape[1]):
        cell_receptors = receptor_expression(receptors, unkVec[17, j], unkVec[20, j], unkVec[19, j], unkVec[21, j])
        pred_data[:, :, j] = calc_dose_response_mutein(unkVec[:, j], tps, muteinC, ligand_name, cell_receptors)
        df_pred = pds.DataFrame(
            {
                "Cells": np.tile(np.array(cell_name), num),
                "Ligand": np.tile(np.array(ligand_name), num),
                "Time Point": np.tile(tps, 12),
                "Concentration": mutein_conc.reshape(num),
                "Activity Type": np.tile(np.array("predicted"), num),
                "Replicate": np.tile(np.array(j + 1), num),
                "Activity": pred_data[:, :, j].reshape(num),
            }
        )
        df = df.append(df_pred, ignore_index=True)

    return df


def catplot_comparison(ax, df, Mut=True):
    """ Construct EC50 catplots for each time point for Different ligands. """
    # set a manual color palette
    sns.set_palette(sns.xkcd_palette(["violet", "goldenrod"]))
    if Mut:
        sns.set_palette(sns.color_palette("husl", 8)[0:5] + [sns.color_palette("husl", 8)[7]])
        df = df.sort_values(by=["Data Type", "Cell Type", "IL", "Time Point"])

    # plot predicted and experimental EC50
    sns.scatterplot(x="Cell Type", y="EC-50", hue="IL", style="Data Type", data=df.loc[df["Time Point"] == 60.0], legend=False, ax=ax)
    ax.set_xlabel("")  # remove "Cell Type" from xlabel
    ax.set_ylabel(r"EC-50 (log$_{10}$[nM])")
    ax.tick_params(axis='x', which='major', rotation=40, labelsize=6.8)

    expp = Line2D([], [], color="black", marker="x", linestyle="None", markersize=6, label="Experimental")
    predd = Line2D([], [], color="black", marker="o", linestyle="None", markersize=6, label="Predicted")
    ax.legend(handles=[expp, predd])


def nllsq_EC50(x0, xdata, ydata):
    """ Performs nonlinear least squares on activity measurements to determine parameters of Hill equation and outputs EC50. """
    lsq_res = least_squares(residuals, x0, args=(xdata, ydata), bounds=([0.0, 0.0, 0.0], [10.0, 10.0, 10 ** 5.0]), jac="3-point")
    return lsq_res.x[0]


def hill_equation(x, x0, solution=0):
    """ Calculates EC50 from Hill Equation. """
    xk = np.power(x / x0[0], x0[1])
    return (x0[2] * xk / (1.0 + xk)) - solution


def residuals(x0, x, y):
    """ Residual function for Hill Equation. """
    return hill_equation(x, x0) - y


def expScaleWT(predSTAT2, predSTAT15, expSTAT2, expSTAT15, rep2=False):
    """Scales data to model predictions. It is assumed here that predictions and data are lined up by concentration"""
    cellGroups = [['NK'], ['CD8+', 'Naive CD8+', 'Mem CD8+'], ['T-reg', 'Naive Treg', 'Mem Treg'], ['T-helper', 'Naive Th', 'Mem Th']]
    iterator = 0
    output2 = np.zeros(expSTAT2.shape)
    output15 = np.zeros(expSTAT15.shape)

    for cellSet in cellGroups:
        subExpSTAT2 = np.reshape(expSTAT2[(iterator * 4): ((iterator + len(cellSet)) * 4)], (len(cellSet), predSTAT2.shape[3], predSTAT2.shape[1]))
        subExpSTAT15 = np.reshape(expSTAT15[(iterator * 4): ((iterator + len(cellSet)) * 4)], (len(cellSet), predSTAT15.shape[3], predSTAT15.shape[1]))
        subPredSTAT2 = predSTAT2[(iterator): ((iterator + len(cellSet))), :, :, :]
        subPredSTAT15 = predSTAT15[(iterator): ((iterator + len(cellSet))), :, :, :]

        subPredSTAT2 = np.swapaxes(subPredSTAT2, 3, 1)
        subPredSTAT2 = np.swapaxes(subPredSTAT2, 3, 2)
        subPredSTAT15 = np.swapaxes(subPredSTAT15, 3, 1)
        subPredSTAT15 = np.swapaxes(subPredSTAT15, 3, 2)

        if rep2 and cellSet == ['CD8+', 'Naive CD8+', 'Mem CD8+']:
            subPredSTAT2 = np.reshape(subPredSTAT2[0, :, :, :], (1, subPredSTAT2.shape[1], subPredSTAT2.shape[2], subPredSTAT2.shape[3]))
            subPredSTAT15 = np.reshape(subPredSTAT15[0, :, :, :], (1, subPredSTAT15.shape[1], subPredSTAT15.shape[2], subPredSTAT15.shape[3]))
            subExpSTAT2 = np.reshape(subExpSTAT2[0, :, :], (1, subExpSTAT2.shape[1], subExpSTAT2.shape[2]))
            subExpSTAT15 = np.reshape(subExpSTAT15[0, :, :], (1, subExpSTAT15.shape[1], subExpSTAT15.shape[2]))

        subExpSTAT2 = np.reshape(subExpSTAT2, (subExpSTAT2.shape[0], subExpSTAT2.shape[1], subExpSTAT2.shape[2], 1))
        subExpSTAT2 = np.tile(subExpSTAT2, (1, 1, 1, subPredSTAT2.shape[3]))
        subExpSTAT2 = subExpSTAT2[:, 0:2, :, :]
        subExpSTAT15 = np.reshape(subExpSTAT15, (subExpSTAT15.shape[0], subExpSTAT15.shape[1], subExpSTAT15.shape[2], 1))
        subExpSTAT15 = np.tile(subExpSTAT15, (1, 1, 1, subPredSTAT15.shape[3]))
        subExpSTAT15 = subExpSTAT2[:, 0:2, :, :]

        subPredSTAT2 = subPredSTAT2[:, 0:2, :, :]
        subPredSTAT15 = subPredSTAT15[:, 0:2, :, :]
        expSTAT = np.vstack((subExpSTAT2, subExpSTAT15))
        predSTAT = np.vstack((subPredSTAT2, subPredSTAT2))

        ravPred = np.ravel(predSTAT)
        ravExp = np.ravel(expSTAT)
        ravPred = ravPred[~np.isnan(ravExp)]
        ravExp = ravExp[~np.isnan(ravExp)]

        slope, intercept, _, _, _ = stats.linregress(ravExp, ravPred)
        output2[(iterator * 4): ((iterator + len(cellSet)) * 4)] = expSTAT2[(iterator * 4): ((iterator + len(cellSet)) * 4)] * slope + intercept
        output15[(iterator * 4): ((iterator + len(cellSet)) * 4)] = expSTAT15[(iterator * 4): ((iterator + len(cellSet)) * 4)] * slope + intercept
        iterator += len(cellSet)

    return output2, output15


def expScaleMut(mutDF, scaleTimes):
    """Scales data to model predictions for muteins"""
    cellGroups = [['NK'], ['CD8+'], ['T-reg', 'Naive Treg', 'Mem Treg'], ['T-helper', 'Naive Th', 'Mem Th']]
    mutGroups = [["F42Q N-Term", "N88D C-term", "R38Q N-term"], ["WT C-term", "V91K C-term"], ["WT N-term"]]
    expArray = np.array([])
    for mutsGroup in mutGroups:
        for cellSet in cellGroups:
            exp_data = mutDF.loc[(mutDF["Cells"].isin(cellSet)) & (mutDF["Ligand"].isin(mutsGroup)) & (mutDF["Activity Type"] == "experimental")]
            expArray = np.array([])
            pred_data = np.array([])
            
            for cell in cellSet:
                for mutLig in mutsGroup:
                    for time in scaleTimes:
                        expArray = np.append(expArray, np.tile(np.ravel(np.array(exp_data.loc[(exp_data["Cells"] == cell) & (mutDF["Ligand"] == mutLig) & (mutDF["Time Point"] == time)].Activity)), 25))
                        pred_data = np.append(pred_data, np.ravel(np.array(mutDF.loc[(mutDF["Cells"] == cell) & (mutDF["Ligand"] == mutLig) & (mutDF["Activity Type"] == "predicted") & (mutDF["Time Point"] == time)].Activity)))

            slope, intercept, _, _, _ = stats.linregress(expArray, pred_data)

            mutDF.loc[(mutDF["Cells"].isin(cellSet)) & (mutDF["Ligand"].isin(mutsGroup)) & (mutDF["Activity Type"] == "experimental"), "Activity"] = np.array(
                mutDF.loc[(mutDF["Cells"].isin(cellSet)) & (mutDF["Ligand"].isin(mutsGroup)) & (mutDF["Activity Type"] == "experimental"), "Activity"]) * slope
            mutDF.loc[(mutDF["Cells"].isin(cellSet)) & (mutDF["Ligand"].isin(mutsGroup)) & (mutDF["Activity Type"] == "experimental"), "Activity"] = np.array(
                mutDF.loc[(mutDF["Cells"].isin(cellSet)) & (mutDF["Ligand"].isin(mutsGroup)) & (mutDF["Activity Type"] == "experimental"), "Activity"]) + intercept

    return mutDF
