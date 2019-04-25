#!/usr/bin/env python3

from ckine.figures.figureCommon import overlayCartoon
import sys
import matplotlib
matplotlib.use('AGG')

fdir = './Manuscript/Figures/'


if __name__ == '__main__':
    nameOut = 'figure' + sys.argv[1]

    exec('from ckine.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    if sys.argv[1] == '1':
        # Overlay Figure 1 cartoon
        overlayCartoon(fdir + 'figure1.svg',
                       './graph_all.svg', 10, 15, scalee=0.35)

    if sys.argv[1] == '3':
        # Overlay Figure 3 cartoon
        overlayCartoon(fdir + 'figure3.svg',
                       './ckine/data/tensor.svg', 35, 30, scalee=0.65)

    print(nameOut + ' is done.')
