## Model-data comparison

![**Figure title.** Figure explanation.](./Figures/figure4.svg){#fig:expData}



## Tensor Factorization Shows Experimental Results

![**Non-negative CP decomposition applied to experimenal pSTAT measurements.** A) $R^2X$ using non-negative CP decomposition plotted against the number of components the tensor was decomposed into. B) Timepoint decomposition plot showing component values against time for each factorization component after decomposing the tensor into 2 components. C) Decomposition plot along the cell type dimension showing the 11 cell type values along each component. D) Ligand decomposition plot along third dimension revealing each ligand according to its shape and concentration reflected by its color.](./Figures/figure5.svg){#fig:expFac}

Given that this generated tensor was obtained through running simulations, we showed that the decomposition techniques can also be applied to experimental measurements. We collected ligand activity data for cells treated with either wild type IL-2 or IL-15 in a similar setup and constructed a tensor. Non-negative CP decomposition was performed on this experimental tensor, and similar cell, time, and ligand decomposition plots are shown in [@Fig:expFac].
