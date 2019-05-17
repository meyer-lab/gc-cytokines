# Methods

All analysis was implemented in Python, and can be found at <https://github.com/meyer-lab/type-I-ckine-model>, release 1.0.

## Model

### Base model

Cytokine binding to receptors was modeled using ordinary differential equations (ODEs). The ligands of interest were six cytokines: IL-2, IL-15, IL-4, IL-7, IL-9, and IL-21. IL-2 and IL-15 each had two private receptors, one being a signaling-deficient α-chain (IL-2Rα and IL-15Rα) and the other being signaling-competent IL-2Rβ. The other four cytokines each had one signaling-competent private receptor (IL-7Rα, IL-9R, IL-4Rα, IL-21Rα). Ligands must bind first to a private receptor and then can dimerize to other private receptors or the common γ~c~ receptor thereafter. Direct binding of ligand to γ~c~ was not included because such a reaction was not observed experimentally [@Voss2428].

Our model accounted for endosomal trafficking, allowing all species to be endocytosed, recycled, and degraded. Each receptor was synthesized and placed on the cell surface at a unique rate. All binding events occuring on the cell surface also occured in the endosome with the same rate parameters. The surface and endosomal ligand concentrations were assumed to be equal for each cytokine. The number densities of each complex and receptor were different between the cell surface and endosome. 

The rate of change for free receptors and complexes were modeled through ODEs. There were 28 ODEs that represented the rate of change for free receptors and complexes on the cell surface, 28 ODEs that represented the rate of change for free receptors and complexes in the endosome, and 6 ODEs that represented the rate of change of ligand concentration. For a complex ($C$) that forms through the binding of ligand ($L$) to free receptor ($R$) with a forward reaction rate ($k_{bnd}$) and a reverse reaction rate ($k_r$), the ODE would be:

$$\delta C/\delta t=(k_{bnd}*L*R)-(k_{r}*C)$$ 

In this example, $\delta C/\delta t$ is in units of $\mathrm{\frac{number}{cell * min}}$, $L$ is in units of concentration (nM), $R$ and $C$ are in number per cell, $k_{bnd}$ is in $\mathrm{\frac{1}{nM * min}}$, and $k_r$ is in $\mathrm{min^{-1}}$. For our model, all of the free receptors and complexes were measured in units of number per cell and all ligands were measured in units of concentration (nM). Due to these unit choices for our species, the rate constants for ligand binding to a free receptors had units of $\mathrm{\frac{1}{nM * min}}$, rate constants for the forward dimerization of free receptor to complex had units of $\mathrm{\frac{cell}{min * number}}$, and the dissociation of a complex into another complex and free receptor had units of $\mathrm{min^{-1}}$.

Cytokine signaling follows the JAK-STAT signaling pathway which is initiated when two JAK subunits come in contact with one another. JAK proteins are found on the intracellular regions of the γ~c~, IL-2Rβ, IL-4Rα, IL-7Rα, IL-9R, and IL-21Rα receptors; therefore all complexes which contained two signaling-competent receptors were deemed to be active species.

We used CVODE to numerically solve our stiff system of ODEs [@hindmarsh2005sundials]. The inputs for our solver were the initial values of each receptor and complex, time, and all unknown rate parameters. To determine the initial values of each receptor and complex, we ran the model for a long time with no cytokine present, allowing the system to reach steady state. An tolerance of $1.5 * 10^{-6}$ was used in solving our ODEs.

We wrote conservation of species and equilibrium unit tests to ensure that all species in our model behaved in a manner consistent with biological intuition. We also wrote unit tests to ensure the functionality and reproducibility of components in the full model.


### Parameters and assumptions

All ligand-receptor binding processes had a forward rate constant of $k_{bnd}$ which was assumed to be on rate of $10^7$ $\mathrm{\frac{1}{M * sec}}$ (0.6 $\mathrm{\frac{1}{nM * min}}$). All forward dimerization reaction rates were represented by $k_{fwd}$. All reverse reaction rates were unique. We used detailed balance to define 8 unknown reverse reaction rate constants. Many of the equilibrium dissociation constants (K~d~) were found in published surface plasmon resonance and isothermal calorimetry experiments.

+--------------------+------------------+--------------------------------------+
| K~d~               | Value            | Reference                            |
+====================+==================+======================================+
| IL-2                                                                         |
+--------------------+------------------+--------------------------------------+
| $k_{1} / k_{bnd}$  | 10 nM            | [@Rickert_receptor_const]            |
+--------------------+------------------+--------------------------------------+
| $k_{2} / k_{bnd}$  | 144 nM           | [@Rickert_receptor_const]            |
+--------------------+------------------+--------------------------------------+
| $k_{10} / k_{fwd}$ | 12 nM            | [@Rickert_receptor_const]            |
+--------------------+------------------+--------------------------------------+
| $k_{11} / k_{fwd}$ | 63 nM            | [@Rickert_receptor_const]            |
+--------------------+------------------+--------------------------------------+
| $k_{5} / k_{fwd}$  | 1.5 nM           | [@Rickert_receptor_const]            |
+--------------------+------------------+--------------------------------------+
| IL-15                                                                        |
+--------------------+------------------+--------------------------------------+
| $k_{13} / k_{bnd}$ | 0.065 nM         | [@Mortier20012006; @DUBOIS2002537]   |
+--------------------+------------------+--------------------------------------+
| $k_{14} / k_{bnd}$ | 438 nM           | [@ring_mechanistic_2012]             |
+--------------------+------------------+--------------------------------------+
| IL-4                                                                         |
+--------------------+------------------+--------------------------------------+
| $k_{32} / k_{bnd}$ | 1.0 nM           | [@Gonnordeaal1253]                   |
+--------------------+------------------+--------------------------------------+
| IL-7                                                                         |
+--------------------+------------------+--------------------------------------+
| $k_{25} / k_{bnd}$ | 59 nM            | [@Walsh_IL7_2012]                    |
+--------------------+------------------+--------------------------------------+
| IL-9                                                                         |
+--------------------+------------------+--------------------------------------+
| $k_{29} / k_{bnd}$ | 0.1 nM           | [@Renauld5690]                       |
+--------------------+------------------+--------------------------------------+
| IL-21                                                                        |
+--------------------+------------------+--------------------------------------+
| $k_{34} / k_{bnd}$ | 0.07 nM          | [@Gonnordeaal1253]                   |
+--------------------+------------------+--------------------------------------+


The rate of endocytosis is quantified by a constant of $k_{endo,a}$ for active complexes and $k_{endo}$ for all other species. The fraction of all endosomal species sent to lysosomes is $f_{sort}$. All endosomal species not sent to lysosomes are recycled back to the cell surface. The rate constants to quantify degradation and recycling are $k_{deg}$ and $k_{rec}$, respectively. There is no autocrine ligand produced by the cells. Receptors can be synthesized by the cells and placed on the cell surface; receptor synthesis rates are specific to each receptor. The volume of the entire endosome was 10 fl and the surface area of the endosome is half the size of the cell surface [@MEYER201525].


### Model fitting

We used Markov chain Monte Carlo to fit the unknown parameters in our model to experimental data obtained by Ring, et al. and Gonnord, et al. [@ring_mechanistic_2012; @Gonnordeaal1253]. Experimental measurements include p-STAT activity under stimulation with varying concentrations of IL-2, IL-15, IL-4, and IL-7 as well as time-course measurements of surface IL-2Rβ upon IL-2 and IL-15 stimulation. YT-1 human NK cells were used for all data-sets involving IL-2 and IL-15. Human PBMCs were used for all data-sets involving IL-4 and IL-7. All YT-1 cell experiments were repeated in the absence of IL-2Rα. All of the data used from the Ring paper can be found in Figure 5 of said paper and all of the Gonnord data can be found in Figure S3 of said paper. Measurements of receptor counts at steady state in Gonnord et al. were used to solve for IL-7Rα, IL-4Rα, and γ~c~ expression rates in human PBMCs. We created functions that calculated p-STAT activity and percent of surface IL-2Rβ under various cellular conditions in order to calculate the residuals between our model predictions and experimental data. 

When fitting our model to the data, we used PyMC model that incorporates Bayesian Statistics to store the likelihood of the model at a given point. The fitting process varied the unknown rate parameters in an attempt to minimize the residuals between our model predictions and experimental data. All unknown rate parameters were assumed to have a lognormal distribution with a standard deviation of 0.1; the only exception to these distributions was $f_{sort}$ which was assumed to have a beta distribution with shape parameters of $\alpha=20$ and $\beta=40$. Executing this fitting process yielded likelihood distributions of each unknown parameter and sum of squared error between model prediction and experimental data at each point of experiemental data. 

### Tensor Generation and Factorization

### Tensor generation

After fitting the model and obtaining the posterior distributions of each unknown parameter, we generated a three-dimensional dataset for the ligand activities over time upon simulating 12 cell types with IL-2, IL-15, or a mutant IL-2Rα affinity-enriched IL-2. The receptor expression rates for IL-2Rα, IL-2Rβ, γ~c~, and IL-15Rα were calculated based on measured surface abundance ([@Fig:tfac]B). The initial ligand concentrations to stimulate the cells matched those from experimental doses ([@Fig:supp5]D). As a result, the final dataset was a tensor of dimensions 100 timepoints by 13 cell types by 3*X ligand values, where X is the number of initial ligand concentrations used for each of IL-2, IL-15, and mutant IL-2. From there, tensor factorization was employed to visualize variation in the data.

### Tensor factorization

Before decomposition, each tensor was variance scaled across cell populations. Tensor factorization analysis was performed using a python package called TensorLy. We applied non-negative Canonical Polyadic Decomposition (also known as CP or PARAFAC) to obtain one matrix (or factor) per mode of the tensor ([@Fig:tfac]A). This decomposition follows the scheme that just as a matrix can be expressed as the sum of the outer product of two vectors, a tensor of order 3 can be expressed as the sum of the ‘R’ outer product of three vectors. Each vector then belongs to a decomposition matrix, called a factor, along the corresponding dimension. From there, each of the three factor matrices will have R columns.
 
In order to determine the number of components (R) necessary to explain 90% of the variance in the original tensor, an $R^2X$ metric was computed and compared for decomposing the tensor using different number of components. This method first required to decompose the tensor into the sum of R vector products utilizing tensorly.parafac, which results in 3 factor matrices whose second dimension is R. Following, the tensor was reconstructed with tensorly.kruskal_to_tensor by multiplying the factors out. Then, the $R^2X$ was computed using

$$R^2X = 1 - (variance(X_r-X)/variance(X)).$$

Here, X_r is the reconstructed tensor and X is the original tensor. By iterating over different possible R component numbers, the percent explained variance is obtained. 
 
The original simulated tensor was then decomposed to 4 components along each dimension and the columns within each factor matrix were plotted against each other to help visualize corresponding relationships between variables. 

We also employed non-negative Tucker Decomposition. This decomposed our tensor into a core tensor and three matrices which correspond to the core scalings along each mode of time, cells, and active ligands. This meant that the tensor’s first dimension (time) was decomposed to 3 components, and the second and third dimensions were decomposed to 4 components each ([@Fig:supp4]A-E). 
