# Methods

All analysis was implemented in Python, and can be found at <https://github.com/meyer-lab/type-I-ckine-model>, release 1.0.

## Model

### Base model

Cytokine binding to receptors was modeled using a kinetic model. The ligands of interest were six cytokines: IL-2, IL-15, IL-4, IL-7, IL-9, and IL-21. Each ligand can bind to the common γ~c~ receptor and a private family of receptors specific to each ligand. IL-2's private receptors were IL-2Rα and IL-2Rβ; IL-15's private receptors were IL-15Rα and IL-2Rβ; IL-4's private receptor was IL-4Rα; IL-7's private receptor was IL-7Rα; IL-9's private receptor was IL-9R; IL-21's private receptor was IL-21Rα. Ligands bind to a private receptor and dimerize with other receptors thereafter to form larger complexes. All receptors were immobilized on the cells, meaning no soluble receptor was present in any compartment. Every complex has exactly one ligand and has at least one receptor. Each of these dimerization steps had forward and reverse reaction rate constants based on the dissociation constants of the complexes involved.

Our model accounted for endosomal trafficking, allowing all species to be endocytosed, recycled, and degraded. Each receptor was synthesized and placed on the cell surface at a unique rate. All binding events occuring on the cell surface also occured in the endosome with the same rate parameters. The surface and endosomal ligand concentrations were assumed to be equal for each cytokine. The number densities of each complex and receptor were different between the cell surface and endosome. 

The rate of change for free receptors and complexes were modeled through ordinary differential equations (ODEs). There were 28 ODEs that represented the rate of change for free receptors and complexes on the cell surface, 28 ODEs that represented the rate of change for free receptors and complexes in the endosome, and 6 ODEs that represented the rate of change of ligand concentration. For a complex ($C$) that forms through the binding of ligand ($L$) to free receptor ($R$) with a forward reaction rate ($k_f$) and a reverse reaction rate ($k_r$), the ODE would be:

$$\delta C/\delta t=(k_{f}*L*R)-(k_{r}*C)$$ 

In this example, $\delta C/\delta t$ is in units of $\mathrm{\frac{number}{cell * min}}$, $L$ is in units of concentration (nM), $R$ is in $\mathrm{\frac{number}{cell}}$, $C$ is in $\mathrm{\frac{number}{cell}}$, $k_f$ is in $\mathrm{\frac{1}{nM * min}}$, and $k_r$ is in $\mathrm{\frac{1}{min}}$. For our model, all of the free receptors and complexes were measured in units of $\mathrm{\frac{number}{cell}}$ and all ligands were measured in units of concentration (nM). Due to these unit choices for our species, the rate constants for ligand binding to a free receptors had units of $\mathrm{\frac{1}{nM * min}}$, rate constants for the forward dimerization of free receptor to complex had units of $\mathrm{\frac{cell}{min * number}}$, and the dissociation of a complex into another complex and free receptor had units of $\mathrm{\frac{1}{min}}$.

Type I cytokine signaling follows the JAK-STAT signaling pathway which is initiated when two JAK subunits come in contact with one another. JAK proteins are found on the intracellular regions of the γ~c~, IL-2Rβ, IL-4Rα, IL-7Rα, IL-9R, and IL-21Rα receptors; therefore all complexes which contained at least two of those receptors were deemed to be active species.

We used CVODE to numerically solve our stiff system of ODEs [@hindmarsh2005sundials]. The inputs for our solver were the initial values of each receptor and complex, time, and all unknown rate parameters. To determine the initial values of each receptor and complex, we ran the model for a long time with no cytokine present, allowing the system to reach steady state. An tolerance of $1.5 * 10^{-6}$ was used in solving our ODEs.

We wrote conservation of species and equilibrium unit tests to ensure that all species in our model behaved in a manner consistent with biological intuition. We also wrote unit tests to ensure the functionality and reproducibility of components in the full model.


### Parameters and assumptions

All ligand-receptor binding processes had a forward rate constant of $k_{bnd}$ which was assumed to be on rate of $10^7$ $\mathrm{\frac{1}{M * sec}}$ (0.6 $\mathrm{\frac{1}{nM * min}}$); the only exception to this assumption was the binding of free ligand to γ~c~, which was assumed to not occur in our model due to weak affinity [@Voss2428]. All forward dimerization reaction rates were represented by $k_{fwd}$. All reverse reaction rates were unique. We used detailed balance to eliminate 8 unknown reverse reaction rate constants. Detailed balance loops were based on formation of full complexes and initial assembly. Each forward and reverse reaction rate was related through an equilibrium dissociation constant (K~d~). Many of these equilibrium dissociation constants were found in published surface plasmon resonance and isothermal calorimetry experiments.


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
| $k_{13} / k_{bnd}$ | 0.065 nM         | multiple papers suggesting 30-100 pM |
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

We used Markov chain Monte Carlo to fit the unknown parameters in our model to experimental data obtained by Ring, et al. and Gonnord, et al. [@ring_mechanistic_2012, @Gonnordeaal1253]. Experimental measurements include p-STAT activity under stimulation with varying concentrations of IL-2, IL-15, IL-4, and IL-7 as well as time-course measurements of total IL-2Rβ located on the cell surface under IL-2 and IL-15 stimulation. YT-1 human NK cells were used for all data-sets involving IL-2 and IL-15. Human PBMCs were used for all data-sets involving IL-4 and IL-7. All YT-1 cell experiments were repeated in the absence of IL-2Rα. All of the data used from the Ring paper can be found in Figure 5 of said paper and all of the Gonnord data can be found in Figure S3 of said paper. Measurements of receptor counts at steady state in Gonnord et al. were used to algebraically solve for IL-7Rα, IL-4Rα and γ~c~ expression levels in human PBMCs. We created functions that calculated p-STAT activity and percent of surface IL-2Rβ under various cellular conditions in order to calculate the residuals between our model predictions and experimental data. 

When fitting our model to the data, we used PyMC model that incorporates Bayesian Statistics to store the likelihood of the model at a given point. The fitting process varied the unknown rate parameters in an attempt to minimize the residuals between our model predictions and experimental data. All unknown rate parameters were assumed to have a lognormal distribution with a standard deviation of 0.1; the only exception to these distributions was $f_{sort}$ which was assumed to have a beta distribution with shape parameters of $\alpha=20$ and $\beta=40$. Executing this fitting process yielded likelihood distributions of each unknown parameter and sum of squared error between model prediction and experimental data at each point of experiemental data. 

We built a function that calculated the Jacobian matrix for all free receptors, ligands, and complexes on the cell surface and in the endosome. The Jacobian matrix was used to aid the process of debugging why certain combinations of parameters and complex concentrations caused our solver to fail. 

### Tensor Generation and Factorization

### Tensor generation

After fitting the model and obtaining the posterior distributions of the unknown parameters, we generated a three-dimensional dataset for the ligand activities at different timepoints upon simulating eight cell types exclusively with one of IL-2, IL-15, and a mutant IL-2Rα affinity-enriched IL-2. The receptor expression rates for IL-2Rα, IL-2Rβ, γ~c~, and IL-15Rα were calculated after measuring their levels on different cells ([@Fig:tfac]B). The initial ligand concentrations to stimulate the cells with ranged from $10^{-2}$ to $10^1$ nM. Each combination of these initial conditions, when inputted into the model, resulted in a matrix of size 100 timepoints by 62 species values. In order to interpret the concentration values at different timepoints, we reduced these values to three, accounting for the respective active species concentrations. As a result, the final dataset was a tensor of dimensions 100 timepoints by 8 cell types by 3*X ligand values, where X is the number of initial ligand concentrations used for each of IL-2, IL-15, and mutant IL-2. From there, further tensor decomposition methods such as tensor factorization were employed in order to achieve intuitive representation and handling of higher dimensional data without losing the structural characteristics of the data itself. 

### Tensor factorization

Before deconstructing the dataset and applying any decomposition methods, the tensor was scaled along the dimension of the 3*X values in order to mitigate the different scales available in the varied results. This z-scoring ensures the data is put on a similar scale during analysis.

Tensor factorization analysis was performed using a python package called TensorLy. Given our tensor of order 3, we applied Canonical Polyadic Decomposition (also known as CP or PARAFAC) to obtain one matrix (or factor) per mode of the tensor ([@Fig:tfac]A). This decomposition follows the scheme that just as a matrix can be expressed as the sum of the outer product of two vectors, a tensor of order 3 can be expressed as the sum of the ‘R’ outer product of three vectors. Each vector then belongs to a decomposition matrix, called a factor, along the corresponding dimension. From there, each of the three factor matrices will have R columns. More specifically, we used the non-negative form of CP decomposition to exclusively inspect individual cytokines as we were not simulating cross-inhibition. 
 
In order to determine the number of components (R) necessary to explain 90% of the variance in the original tensor, an $R^2X$ metric was computed and compared for decomposing the tensor using different number of components. This method first required to decompose the tensor into the sum of R vector products utilizing tensorly.parafac, which results in 3 factor matrices whose second dimension is R. Following, the tensor was reconstructed with tensorly.kruskal_to_tensor by multiplying the factors out. Then, the $R^2X$ was computed using

$$R^2X = 1 - (variance(X_r-X)/variance(X)).$$

Here, X_r is the reconstructed tensor and X is the original tensor. By iterating over different possible R component numbers, the percent explained variance is obtained. 
 
From the $R^2X$ values, we determined that 3 components were needed to explain 90% of the variance ([@Fig:tfac]C). Yet we used 4 for plotting purposes. The original tensor was then decomposed to 4 components along each dimension and the columns within each factor matrix were plotted against each other to help visualize corresponding relationships between variables. 

Moreover, we employed another tensor decomposition method called Tucker Decomposition through its non-negative form. This decomposed our tensor into a core tensor and three matrices which correspond to the core scalings along each mode of time, cells, and active ligands. Since Tucker decomposition is advised for data compression and dimensionality reduction, we kept an $R^2X$ of 99% when choosing the number of components. This meant that the tensor’s first dimension (time) was decomposed to 3 components, and the second and third dimensions were decomposed to 4 components each ([@Fig:supp4]A-E). 
