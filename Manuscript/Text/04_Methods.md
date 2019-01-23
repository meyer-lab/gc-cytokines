# Methods

All analysis was implemented in Python, and can be found at <https://github.com/meyer-lab/type-I-ckine-model>, release 1.0.

## Model

### Base model

Cytokine binding to receptors was modeled using a kinetic model. The ligands of interest were six cytokines: IL-2, IL-15, IL-4, IL-7, IL-9, and IL-21. Each ligand can bind to the common γ~c~ receptor and a private family of receptors specific to each ligand. IL-2's private receptors were IL-2Rα and IL-2Rβ; IL-15's private receptors were IL-15Rα and IL-2Rβ; IL-4's private receptor was IL-4Rα; IL-7's private receptor was IL-7Rα; IL-9's private receptor was IL-9R; IL-21's private receptor was IL-21Rα. Ligands bind to a private receptor and dimerize with other receptors thereafter to form larger complexes. All receptors were immobilized on the cells, meaning no soluble receptor was present in any compartment. Every complex has exactly one ligand and has at least one receptor. Each of these dimerization steps had forward and reverse reaction rate constants based on the dissociation constants of the complexes involved.

Our model accounted for endosomal trafficking, allowing all species to be endocytosed, recycled, and degraded. Each receptor was synthesized and placed on the cell surface at a unique rate. All binding events occuring on the cell surface also occured in the endosome with the same rate parameters. The surface and endosomal ligand concentrations were assumed to be equal for each cytokine. The number densities of each complex and receptor were different between the cell surface and endosome. 

The rate of change for free receptors and complexes were modeled through ordinary differential equations (ODEs). There were 28 ODEs that represented the rate of change for free receptors and complexes on the cell surface, 28 ODEs that represented the rate of change for free receptors and complexes in the endosome, and 6 ODEs that represented the rate of change of ligand concentration. For a complex ($C$) that forms through the binding of ligand ($L$) to free receptor ($R$) with a forward reaction rate ($k_f$) and a reverse reaction rate ($k_r$), the ODE would be:

$$dC/dt=(k_{f}*L*R)-(k_{r}*C)$$ 

In this example, $\delta C/\delta t$ is in units of number / cell * min, $L$ is in units of molarity, $R$ is in number / cell, $C$ is in number / cell, $k_f$ is in 1 / nM * min, and $k_r$ is in 1 / min. For our model, all of the free receptors and complexes were measured in units of number per cell and all ligands were measured in units of concentration (nM). Due to these unit choices for our species, the rate constants for ligand binding to a free receptors had units of 1 / nM * min, rate constants for the forward dimerization of free receptor to complex had units of 1 / min * number per cell, and the dissociation of a complex into another complex and free receptor had units of 1 / min.

Type I cytokine signaling follows the JAK-STAT signaling pathway which is initiated when two JAK subunits come in contact with one another. JAK proteins are found on the intracellular regions of the γ~c~, IL-2Rβ, IL-4Rα, IL-7Rα, IL-9R, and IL-21Rα receptors; therefore all complexes which contained at least two of those receptors were deemed to be active species.

We used CVODE to numerically solve our stiff system of ODEs [@hindmarsh2005sundials]. The inputs for our solver were the initial values of each receptor and complex, time, and all unknown rate parameters. To determine the initial values of each receptor and complex, we ran the model for a long time with no cytokine present, allowing the system to reach steady state. An tolerance of 1.5E-6 was used in solving our ODEs.

We wrote conservation of species and equilibrium unit tests to ensure that all species in our model behaved in a manner consistent with biological intuition. We also wrote unit tests to ensure the functionality and reproducibility of components in the full model.


### Parameters and assumptions

All ligand-receptor binding processes had a forward rate constant of $k_{bnd}$ which was assumed to be on rate of $10^7$ M-1 sec-1 (0.6 nM-1 min-1); the only exception to this assumption was the binding of free ligand to γ~c~, which was assumed to not occur in our model due to weak affinity [@Voss2428]. All forward dimerization reaction rates were represented by $k_{fwd}$. All reverse reaction rates were unique. We used detailed balance to eliminate 8 unknown reverse reaction rate constants. Detailed balance loops were based on formation of full complexes and initial assembly. Each forward and reverse reaction rate was related through an equilibrium dissociation constant ($K_d$). Many of these equilibrium dissociation constants were found in published surface plasmon resonance and isothermal calorimetry experiments.


+-----------------+------------------+--------------------------------------+
| $K_D$           | Value            | Reference                            |
+=================+==================+======================================+
| IL-2                                                                      |
+-----------------+------------------+--------------------------------------+
| $k1 / k_{bnd}$  | 10 nM            | [@Rickert_receptor_const]            |
+-----------------+------------------+--------------------------------------+
| $k2 / k_{bnd}$  | 144 nM           | [@Rickert_receptor_const]            |
+-----------------+------------------+--------------------------------------+
| $k10 / k_{fwd}$ | 12 nM            | [@Rickert_receptor_const]            |
+-----------------+------------------+--------------------------------------+
| $k11 / k_{fwd}$ | 63 nM            | [@Rickert_receptor_const]            |
+-----------------+------------------+--------------------------------------+
| $k5 / k_{fwd}$  | 1.5 nM           | [@Rickert_receptor_const]            |
+-----------------+------------------+--------------------------------------+
| IL-15                                                                     |
+-----------------+------------------+--------------------------------------+
| $k13 / k_{bnd}$ | 0.065 nM         | multiple papers suggesting 30-100 pM |
+-----------------+------------------+--------------------------------------+
| $k14 / k_{bnd}$ | 438 nM           | [@ring_mechanistic_2012]             |
+-----------------+------------------+--------------------------------------+
| IL-4                                                                      |
+-----------------+------------------+--------------------------------------+
| $k32 / k_{bnd}$ | 1.0 nM           | [@Gonnordeaal1253]                   |
+-----------------+------------------+--------------------------------------+
| IL-7                                                                      |
+-----------------+------------------+--------------------------------------+
| $k25 / k_{bnd}$ | 59 nM            | [@Walsh_IL7_2012]                    |
+-----------------+------------------+--------------------------------------+
| IL-9                                                                      |
+-----------------+------------------+--------------------------------------+
| $k29 / k_{bnd}$ | 0.1 nM           | [@Renauld5690]                       |
+-----------------+------------------+--------------------------------------+
| IL-21                                                                     |
+-----------------+------------------+--------------------------------------+
| $k34 / k_{bnd}$ | 0.07 nM          | [@Gonnordeaal1253]                   |
+-----------------+------------------+--------------------------------------+


The rate of endocytosis is quantified by a constant of $activeEndo$ for active complexes and $endo$ for all other species. The fraction of all endosomal species sent to lysosomes is $f_{sort}$. All endosomal species not sent to lysosomes are recycled back to the cell surface. The rate constants to quantify degradation and recycling are $k_{Deg}$ and $k_{Rec}$, respectively. There is no autocrine ligand produced by the cells. Receptors can be synthesized by the cells and placed on the cell surface; receptor synthesis rates are specific to each receptor. The volume of the entire endosome was 10 fl and the surface area of the endosome is half the size of the cell surface [@MEYER201525].


### Model fitting

We used Markov chain Monte Carlo to fit the unknown parameters in our model to experimental data obtained by Ring, et al. and Gonnord, et al. [@ring_mechanistic_2012, @Gonnordeaal1253]. Experimental measurements include p-STAT activity under stimulation with varying concentrations of IL-2, IL-15, IL-4, and IL-7 as well as time-course measurements of total IL-2Rβ located on the cell surface under IL-2 and IL-15 stimulation. YT-1 human NK cells were used for all data-sets involving IL-2 and IL-15. Human PBMCs were used for all data-sets involving IL-4 and IL-7. All YT-1 cell experiments were repeated in the absence of IL-2Rα. All of the data used from the Ring paper can be found in Figure 5 of said paper and all of the Gonnord data can be found in Figure S3 of said paper. Measurements of receptor counts at steady state in Gonnord et al. were used to algebraically solve for IL-7Rα, IL-4Rα and γ~c~ expression levels in human PBMCs. We created functions that calculated p-STAT activity and percent of surface IL-2Rβ under various cellular conditions in order to calculate the residuals between our model predictions and experimental data. 

When fitting our model to the data, we used PyMC model that incorporates Bayesian Statistics to store the likelihood of the model at a given point. The fitting process varied the unknown rate parameters in an attempt to minimize the residuals between our model predictions and experimental data. All unknown rate parameters were assumed to have a lognormal distribution with a standard deviation of 0.1; the only exception to these distributions was $f_{sort}$ which was assumed to have a beta distribution with shape parameters of $\alpha=20$ and $\beta=40$. Executing this fitting process yielded likelihood distributions of each unknown parameter and sum of squared error between model prediction and experimental data at each point of experiemental data. 

We built a function that calculated the Jacobian matrix for all free receptors, ligands, and complexes on the cell surface and in the endosome. The Jacobian matrix was used to aid the process of debugging why certain combinations of parameters and complex concentrations caused our solver to fail. 

### Tensor Generation and Factorization

### Tensor generation

After fitting the model and obtaining the posterior distributions of the unknown parameters, we generated a multi-dimensional dataset for the receptor-complex concentrations at different timepoints for various combinations of receptor expression rates and initial ligand concentrations. The receptor expression rates are those of IL-2Rα, IL-2Rβ, γ~c~, IL-15Rα, IL-7Rα, and IL-9R, and they range from $10^{-3}$ to $10^2$ nM/min. The initial ligand concentrations to stimulate the cells with include IL-2, IL-15, IL-7, and IL-9, and they cover from $10^{-3}$ to $10^3$ nM. Each combination of these initial conditions, when inputted into the model, results in a matrix of size 1000 timepoints by 56 concentration values. In order to interpret the concentration values at different timepoints, we reduced the 56 values to 16, four of which include the amount of surface and endosomal active species per cytokine, six include the surface receptor concentrations, and another six encompass the total receptor amounts (both surface and endosomal). As a result, from running X combinations, the final dataset is a tensor of dimensions X combinations by 1000 timepoints by 16 values. From there, further tensor decomposition methods such as tensor factorization were employed in order to achieve intuitive representation and handling of higher dimensional data without losing the structural characteristics of the data itself. 

### Tensor factorization

Before deconstructing the dataset and applying any decomposition methods, the tensor was scaled along the dimension of the 16 values in order to mitigate the different scales available in the varied results. This z-scoring ensures the data is put on a similar scale during analysis.

Tensor factorization analysis was performed using a python package called TensorLy. Given our tensor of order 3, we applied Canonical Polyadic Decomposition (also known as CP or PARAFAC) to obtain one matrix (or factor) per mode of the tensor. This decomposition follows the scheme that just as a matrix can be expressed as the sum of the outer product of two vectors, a tensor of order 3 can be expressed as the sum of the ‘R’ outer product of three vectors. Each vector then belongs to a decomposition matrix, called a factor, along the corresponding dimension. From there, each of the three factor matrices will have R columns. 
 
In order to determine the number of components (R) necessary to explain 90 percent of the variance in the original tensor, an R2X metric was computed and compared for decomposing the tensor using different number of components. This method first required to decompose the tensor into the sum of R vector products utilizing tensorly.parafac, which results in 3 factor matrices whose second dimension is R. Following, the tensor was reconstructed with tensorly.kruskal_to_tensor by multiplying the factors out. Then, the R2X was computed using

$$R^2X = 1 - (variance(X_r-X)/variance(X)).$$

Here, Xr is the reconstructed tensor and X is the original tensor. By iterating over different possible R component numbers, the percent explained variance is obtained. 
 
From the R2X values, we determined that 8 components were needed to explain 90 percent of the variance. The original tensor was then decomposed to 8 components and the columns within each factor matrix were plotted against each other to help visualize correlation and relationships between variables. 

### Correlation Coefficients

As mentioned above, we generated the tensor using 10 input variables. To calculate the correlation coefficients between these input variables and the 8 components to which the tensor was decomposed to, we use the Pearson product-moment correlation coefficient. This is done through numpy.corrcoef between all possible values of each input, i.e. IL-2Rα, IL-2Rβ, γ~c~, IL-15Rα, IL-7Rα, and IL-9R expression rates and IL-2, IL-15, IL-7, and IL-9 ligand concentrations, and each column of the combination decomposition factor matrix. 

