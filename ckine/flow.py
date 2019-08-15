"""
This file includes various methods for flow cytometry analysis.
"""
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from FlowCytometryTools import FCMeasurement
from FlowCytometryTools import QuadGate
import sklearn
from sklearn.decomposition import PCA

# Please note: if trying to display plots in jupyternotebook, must use: %matplotlib inline


def importF(pathname):
    """Import FCS files. Variable input: name of path name to file. Output is a list of Data File Names in FCT Format"""
    # Declare arrays and int
    file = []
    sample = []
    z = 0
    # Read in user input for file path and assign to array file
    pathlist = Path(r"" + str(pathname)).glob("**/*.fcs")
    for path in pathlist:
        path_in_str = str(path)
        file.append(path_in_str)
    # Go through each file and assign the file contents to entry in the array sample
    for entry in file:
        sample.append(FCMeasurement(ID="Test Sample" + str(z), datafile=entry))
        z += 1
    importF.sample = sample
    # Returns the array sample which contains data of each file in folder (one file per entry in array)
    return sample


# *********************************** Gating Fxns *******************************************
# Treg and NonTreg

# add which channels relate to the proteins
def treg():
    """Function for creating and returning the T reg gate"""
    # T reg: Take quad gates for T reg cells and combine them to create single, overall T reg gate
    treg1 = QuadGate((7.063e03, 3.937e03), ("BL1-H", "VL1-H"), region="top left", name="treg1")
    treg2 = QuadGate((5.412e03, 6.382e03), ("BL1-H", "VL1-H"), region="bottom right", name="treg2")
    treg_gate = treg1 & treg2
    return treg_gate


def nonTreg():
    """Function for creating and returning the Non T reg gate"""
    # non T reg: Take quad gates for non T reg cells and combine to create overall non T reg gate
    nonTreg1 = QuadGate((5.233e03, 1.731e03), ("BL1-H", "VL1-H"), region="top left", name="nonTreg1")
    nonTreg2 = QuadGate((2.668e03, 5.692e03), ("BL1-H", "VL1-H"), region="bottom right", name="nonTreg2")
    nonTreg_gate = nonTreg1 & nonTreg2
    return nonTreg_gate


def nk():
    """Function for creating and returning the NK gate"""
    # NK cells: Take quad gates for NK cells and combine them to create single, overall NK gate
    nk1 = QuadGate((6.468e03, 4.861e03), ("BL1-H", "VL4-H"), region="top left", name="nk1")
    nk2 = QuadGate((5.550e03, 5.813e03), ("BL1-H", "VL4-H"), region="bottom right", name="nk2")
    nk_gate = nk1 & nk2
    return nk_gate


def bnk():
    """Function for creating and returning the BNK gate"""
    # Bright NK cells: Take quad gates for bright NK cells and combine them to create single, overall bright NK gate
    bnk1 = QuadGate((7.342e03, 4.899e03), ("BL1-H", "VL4-H"), region="top left", name="bnk1")
    bnk2 = QuadGate((6.533e03, 5.751e03), ("BL1-H", "VL4-H"), region="bottom right", name="bnk2")
    bnk_gate = bnk1 & bnk2
    return bnk_gate


def cd():
    """Function for creating and returning the CD gate"""
    # CD cells: Take quad gates for CD cells and combine them to create single, overall CD gate
    cd1 = QuadGate((9.016e03, 5.976e03), ("RL1-H", "VL4-H"), region="top left", name="cd1")
    cd2 = QuadGate((6.825e03, 7.541e03), ("RL1-H", "VL4-H"), region="bottom right", name="cd2")
    cd_gate = cd1 & cd2
    return cd_gate


def cellCount(sample_i, gate):
    """
    Function for returning the count of cells in a single .fcs. file of a single cell file. Arguments: single sample/.fcs file and the gate of the
    desired cell output.
    """
    # Import single file and save data to a variable --> transform to logarithmic scale
    smpl = sample_i.transform("hlog", channels=["BL1-H", "VL1-H", "VL4-H", "RL1-H"])
    # Apply T reg gate to overall data --> i.e. step that detrmines which cells are T reg
    cells = smpl.gate(gate)
    # Number of events (AKA number of cells)
    cell_count = cells.get_data().shape[0]
    # print('Number of Treg cells:' + str(treg_count))
    return cell_count


def rawData(sample_i, gate):
    """
    Function that returns the raw data of certain cell population in a given file. Arguments: sample_i is a single entry/.fcs file and the gate
    of the desired cell population.
    """
    smpl = sample_i.transform("hlog", channels=["BL1-H", "VL1-H", "VL4-H", "RL1-H"])
    # Apply T reg gate to overall data --> i.e. step that detrmines which cells are T reg
    cells = smpl.gate(gate)
    # Get raw data of t reg cells in file
    cell_data = cells.get_data()
    return cell_data


def tcells(sample_i, treg_gate, nonTreg_gate, i):
    """
    Function that is used to plot the Treg and NonTreg gates. Treg (yellow) and Non Treg (green). sample_i is an indivual flow cytommetry
    file/data.
    """
    # Assign data of current file for analysis to variable smpl and transform to log scale
    smpl = sample_i.transform("hlog", channels=["BL1-H", "VL1-H"])
    # CD25 v. Foxp33: VL1 v. BL1
    # Treg
    # Apply T reg gate to overall data --> step that determines which cells are Treg
    treg_cells = smpl.gate(treg_gate)
    # Non Tregs
    # Apply non T reg gate to overall data --> step that detrmines which cells are non T reg
    nonTreg_cells = smpl.gate(nonTreg_gate)

    # Declare figure and axis
    _, ax = plt.subplots() # ???????? unused fig, but need to leave there to make sure right assignment for ax?
    # Plot the treg gate
    treg_cells.plot(["BL1-H", "VL1-H"], color="y")
    # Plot the non Treg gate
    nonTreg_cells.plot(["BL1-H", "VL1-H"], color="g")
    # Plot all of the cells in the file
    print(i)
    ax.set_title("T Reg + Non T Reg - Gating - File " + str(i), fontsize=20)
    smpl.plot(["BL1-H", "VL1-H"])

    bar_T = ax.bar(np.arange(0, 10), np.arange(1, 11), color="y")
    bar_NT = ax.bar(np.arange(0, 10), np.arange(30, 40), bottom=np.arange(1, 11), color="g")
    ax.legend([bar_T, bar_NT], ("T Reg", "Non T Reg"), loc="upper left")


def nk_bnk_plot(sample_i, nk_gate, bnk_gate, i):
    """
    Function that plots the graph of NK and Bright NK cells (both are determined by same x, y-axis). Arguemnt 1: current sample (a single file).
    Argument 2: the gate for NK. Argument 3: the gate for bright NK.
    """
    smpl = sample_i.transform("hlog", channels=["BL1-H", "VL4-H", "RL1-H"])

    # CD3 v. CD56: VL4 v. BL1
    # NK
    # Apply NK gate to overall data --> step that determines which cells are NK
    nk_cells = smpl.gate(nk_gate)
    # CD56 Bright NK
    # Apply Bright NK gate to overall data --> step that determines which cells are Bright NK
    bnk_cells = smpl.gate(bnk_gate)

    _, ax1 = plt.subplots()
    ax1.set_title("CD56 BrightNK + NK - Gating - File " + str(i), fontsize=20)
    nk_cells.plot(["BL1-H", "VL4-H"], color="y", label="NK")
    bnk_cells.plot(["BL1-H", "VL4-H"], color="g", label="Bright NK")
    smpl.plot(["BL1-H", "VL4-H"])

    bar_NK = ax1.bar(np.arange(0, 10), np.arange(1, 11), color="y")
    bar_BNK = ax1.bar(np.arange(0, 10), np.arange(30, 40), bottom=np.arange(1, 11), color="g")
    ax1.legend([bar_NK, bar_BNK], ("NK", "Bright NK"), loc="upper left")


def cd_plot(sample_i, cd_gate, i):
    """
    Function that plots the graph of CD cells. Argument 1: current sample (a single file). Argument 2: the gate for CD cells. Argument 3: the value
    of the current i in a for loop --> use
    when plotting multiple files.
    """
    smpl = sample_i.transform("hlog", channels=["BL1-H", "VL4-H", "RL1-H"])
    # CD3 v. CD8: VL4 v. RL1
    # CD3+CD8+
    # Apply CD cell gate to overall data --> step that determines which cells are CD
    cd_cells = smpl.gate(cd_gate)

    _, ax2 = plt.subplots()
    ax2.set_title("CD3+CD8+ - Gating - File " + str(i), fontsize=20)
    cd_cells.plot(["RL1-H", "VL4-H"], color="b")
    smpl.plot(["RL1-H", "VL4-H"])

    bar_CD = ax2.bar(np.arange(0, 10), np.arange(1, 11), color="b")
    ax2.legend([bar_CD], ("CD3+8+"), loc="upper left")


def count_data(sampleType, gate):
    """
    Used to count the number of cells and store the data of all of these cells in a folder with multiple files --> automates the process sampleType
    is NK or T cell data, gate is the desired cell population.
    Sample type: is the overall importF assignment for T or NK (all the T cell files, all NK cell files)
    """
    # declare the arrays to store the data
    count_array = []
    data_array = []
    # create the for loop to file through the data and save to the arrays
    # using the functions created above for a singular file
    for _, sample in enumerate(sampleType):
        count_array.append(cellCount(sample, gate))
        data_array.append(rawData(sample, gate))
    # returns the array for count of cells and the array where each entry is the data for the specific cell population in that .fcs file
    return count_array, data_array


def plotAll(sampleType, check, gate1, gate2):
    """
    Ask the user to input 't' for t cell, 'n' for nk cell, and 'c' for cd cell checks are used to determine if user input a T-cell, NK-cell, or
    CD-cell gate automates the process for plotting multiple files.
    """
    if check == "t":
        for i, sample in enumerate(sampleType):
            tcells(sample, gate1, gate2, i)
    elif check == "n":
        for i, sample in enumerate(sampleType):
            nk_bnk_plot(sample, gate1, gate2, i)
    elif check == "c":
        for i, sample in enumerate(sampleType):
            cd_plot(sample, gate1, i)


# ********************************** PCA Functions****************************************************


def sampleT(smpl):
    """Output is the T cells data (the protein channels related to T cells)"""
    # Features are the protein channels of interest when analyzing T cells
    features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    # Transform to put on log scale
    tform = smpl.transform("hlog", channels=["BL1-H", "VL1-H", "VL4-H", "BL3-H", "RL1-H"])
    # Save the data of each column of the protein channels
    data = tform.data[["BL1-H", "VL1-H", "VL4-H", "BL3-H"]][0:]
    # Save pSTAT5 data
    pstat = tform.data[["RL1-H"]][0:]
    return data, pstat, features


def sampleNK(smpl):
    """Output is the NK cells data (the protein channels related to NK cells)"""
    # For NK, the data consists of different channels so the data var. output will be different
    # Output is data specific to NK cells
    # Features for the NK file of proteins (CD3, CD8, CD56)
    features = ["VL4-H", "RL1-H", "BL1-H"]
    # Transform all proteins (including pSTAT5)
    tform = smpl.transform("hlog", channels=["VL4-H", "RL1-H", "BL1-H", "BL2-H"])
    # Assign data of three protein channels AND pSTAT5
    data = tform.data[["VL4-H", "RL1-H", "BL1-H"]][0:]
    pstat = tform.data[["BL2-H"]][0:]
    return data, pstat, features


def appPCA(data, features):
    """Applies the PCA algorithm to the data set"""
    # Apply PCA to the data set
    # setting values of data of selected features to data frame
    xi = data.loc[:, features].values
    # STANDARDIZE DATA --> very important to do before applying machine learning algorithm
    xs = sklearn.preprocessing.scale(xi)
    xs = np.nan_to_num(xs)
    # setting how many components wanted --> PC1 and PC2
    pca = PCA(n_components=2)
    # apply PCA to standardized data set
    # NOTE: score == xf
    xf = pca.fit(xs).transform(xs)
    # creates the loading array (equation is defintion of loading)
    loading = pca.components_.T
    return xf, loading


def pcaPlt(xf, pstat, features, i):
    """
    Used to plot the score graph.
    Scattered point color gradients are based on range/abundance of pSTAT5 data. Light --> Dark = Less --> More Active
    """
    # PCA
    if len(features) == 4:
        name = "T Cells"
    elif len(features) == 3:
        name = "NK Cells"
    # Setting x and y values from xf
    x = xf[:, 0]
    y = xf[:, 1]
    # Working with pSTAT5 data --> setting min and max values
    pstat_data = pstat.values

    # Creating a figure for both scatter and mesh plots for PCA
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title(name + " - PCA - File " + str(i), fontsize=20)
    ax.set(xlim=(-5, 5), ylim=(-5, 5))

    # This is the scatter plot of the cell clusters colored by pSTAT5 data
    # lighter --> darker = less --> more pSTAT5 present
    plt.scatter(x, y, s=0.1, c=np.squeeze(pstat_data), cmap="Greens")

def loadingPlot(loading, features, i):
    """Plot the loading data"""
    # Loading
    # Create graph for loading values
    x_load = loading[:, 0]
    y_load = loading[:, 1]

    # Create figure for the loading plot
    fig1 = plt.figure(figsize=(8, 8))
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    plt.scatter(x_load, y_load)

    for z, feature in enumerate(features):
        # Please note: not the best logic, but there are three features in NK and four features in T cells
        if len(features) == 4:
            name = "T Cells"
            if feature == "BL1-H":
                feature = "Foxp3"
            elif feature == "VL1-H":
                feature = "CD25"
            elif feature == "VL4-H":
                feature = "CD4"
            elif feature == "BL3-H":
                feature = "CD45RA"
        if len(features) == 3:
            name = "NK Cells"
            if feature == "VL4-H":
                feature = "CD3"
            if feature == "RL1-H":
                feature = "CD8"
            if feature == "BL1-H":
                feature = "CD56"
        plt.annotate(str(feature), xy=(x_load[z], y_load[z]))
    ax.set_title(name + " - Loading - File " + str(i), fontsize=20)


def pcaAll(sampleType, check):
    """
    Use to plot the score and loading graphs for PCA. Assign protein and pstat5 arrays AND score and loading arrays
    This is all the data for each file.
    Want to use for both T and NK cells? Use it twice!
    sampleType is importF for T or NK
    check == "t" for T cells OR check == "n" for NK cells
    """
    # declare the arrays to store the data
    data_array = []
    pstat_array = []
    xf_array = []
    loading_array = []
    # create the for loop to file through the data and save to the arrays
    # using the functions created above for a singular file
    if check == "t":
        for i, sample in enumerate(sampleType):
            data, pstat, features = sampleT(sample)
            data_array.append(data)
            pstat_array.append(pstat)
            xf, loading = appPCA(data, features)
            xf_array.append(xf)
            loading_array.append(loading)
            pcaPlt(xf, pstat, features, i)
            loadingPlot(loading, features, i)
    elif check == "n":
        for i, sample in enumerate(sampleType):
            data, pstat, features = sampleNK(sample)
            data_array.append(data)
            pstat_array.append(pstat)
            xf, loading = appPCA(data, features)
            pcaPlt(xf, pstat, features, i)
            loadingPlot(loading, features, i)
    return data_array, pstat_array, xf_array, loading_array
