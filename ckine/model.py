import numpy as np

try:
    from numba import jit
except ImportError:
    print('Numba not available, so no JIT compile.')

    def jit(ob):
        return ob


@jit
def dy_dt(y, t, IL2, IL15, IL7, IL9, kfwd, k5rev, k6rev, k15rev, k17rev, k18rev, k22rev, k23rev, k26rev, k27rev, k29rev, k30rev, k31rev):
    # IL2 in nM
    IL2Ra = y[0]
    IL2Rb = y[1]
    gc = y[2]
    IL2_IL2Ra = y[3]
    IL2_IL2Rb = y[4]
    IL2_gc = y[5]
    IL2_IL2Ra_IL2Rb = y[6]
    IL2_IL2Ra_gc = y[7]
    IL2_IL2Rb_gc = y[8]
    IL2_IL2Ra_IL2Rb_gc = y[9]
    
    # IL15 in nM
    IL15Ra = y[10]
    IL15_IL15Ra = y[11]
    IL15_IL2Rb = y[12]
    IL15_gc = y[13]
    IL15_IL15Ra_IL2Rb = y[14]
    IL15_IL15Ra_gc = y[15]
    IL15_IL2Rb_gc = y[16]
    IL15_IL15Ra_IL2Rb_gc = y[17]
    
    #IL7 in nM
    IL7Ra = y[18]
    IL7Ra_IL7 = y[19]
    gc_IL7 = y[20]
    IL7Ra_gc_IL7 = y[21]
    # k25 - k28

    #IL9 in nM
    IL9R = y[22]
    IL9R_IL9 = y[23]
    gc_IL9 = y[24]
    IL9R_gc_IL9 = y[25]
    # k29 - k32

    # These are probably measured in the literature
    kfbnd = 0.01 # Assuming on rate of 10^7 M-1 sec-1
    k1rev = kfbnd * 10 # doi:10.1016/j.jmb.2004.04.038, 10 nM

    k2rev = kfbnd * 144 # doi:10.1016/j.jmb.2004.04.038, 144 nM
    k3fwd = kfbnd / 10.0 # Very weak, > 50 uM. Voss, et al (1993). PNAS. 90, 2428–2432.
    k3rev = 50000 * k3fwd
    k10rev = 12.0 * k5rev * kfwd / 1.5 / kfwd # doi:10.1016/j.jmb.2004.04.038
    k11rev = 63.0 * k5rev * kfwd / 1.5 / kfwd # doi:10.1016/j.jmb.2004.04.038
    
    # Literature values for k values for IL-15
    k13rev = kfbnd * 0.065 #based on the multiple papers suggesting 30-100 pM
    k14rev = kfbnd * 438 # doi:10.1038/ni.2449, 438 nM
    
    # Literature values for IL-7
    k25rev = kfbnd * 59. # DOI:10.1111/j.1600-065X.2012.01160.x, 59 nM
    
    # To satisfy detailed balance these relationships should hold
    # _Based on initial assembly steps
    k4rev = kfbnd * kfwd * k6rev * k3rev / k1rev / kfwd / k3fwd
    k7rev = k3fwd * kfwd * k2rev * k5rev / kfbnd / kfwd / k3rev
    k12rev = kfbnd * kfwd * k1rev * k11rev / kfwd / kfbnd / k2rev
    # _Based on formation of full complex
    k9rev = k2rev * k10rev * k12rev / kfbnd / kfwd / kfwd / k3rev / k6rev * k3fwd * kfwd * kfwd
    k8rev = k2rev * k10rev * k12rev / kfbnd / kfwd / kfwd / k7rev / k3rev * k3fwd * kfwd * kfwd

    # IL15
    # To satisfy detailed balance these relationships should hold
    # _Based on initial assembly steps
    k16rev = kfbnd * kfwd * k18rev * k15rev / k13rev / kfbnd / kfbnd
    k19rev = kfbnd * kfwd * k14rev * k17rev / kfbnd / kfbnd / k15rev
    k24rev = kfbnd * kfwd * k13rev * k23rev / kfwd / kfbnd / k14rev
    # _Based on formation of full complex
    k21rev = k14rev * k22rev * k24rev / kfbnd / kfwd / kfwd / k15rev / k18rev * kfbnd * kfbnd * kfwd
    k20rev = k14rev * k22rev * k24rev / kfbnd / kfwd / kfwd / k19rev / k15rev * kfbnd * kfwd * kfwd

    # _One detailed balance IL7/9 loop
    k32rev = k29rev * k31rev * kfwd * kfbnd / kfbnd / kfwd / k30rev
    k28rev = k25rev * k27rev * kfwd * kfbnd / kfbnd / kfwd / k26rev

    dydt = y.copy()
    
    # IL2
    dydt[0] = -kfbnd * IL2Ra * IL2 + k1rev * IL2_IL2Ra - kfwd * IL2Ra * IL2_gc + k6rev * IL2_IL2Ra_gc - kfwd * IL2Ra * IL2_IL2Rb_gc + k8rev * IL2_IL2Ra_IL2Rb_gc - kfwd * IL2Ra * IL2_IL2Rb + k12rev * IL2_IL2Ra_IL2Rb
    dydt[1] = -kfbnd * IL2Rb * IL2 + k2rev * IL2_IL2Rb - kfwd * IL2Rb * IL2_gc + k7rev * IL2_IL2Rb_gc - kfwd * IL2Rb * IL2_IL2Ra_gc + k9rev * IL2_IL2Ra_IL2Rb_gc - kfwd * IL2Rb * IL2_IL2Ra + k11rev * IL2_IL2Ra_IL2Rb
    dydt[2] = -k3fwd * IL2 * gc + k3rev * IL2_gc - kfwd * IL2_IL2Rb * gc + k5rev * IL2_IL2Rb_gc - kfwd * IL2_IL2Ra * gc + k4rev * IL2_IL2Ra_gc - kfwd * IL2_IL2Ra_IL2Rb * gc + k10rev * IL2_IL2Ra_IL2Rb_gc
    dydt[3] = -kfwd * IL2_IL2Ra * IL2Rb + k11rev * IL2_IL2Ra_IL2Rb - kfwd * IL2_IL2Ra * gc + k4rev * IL2_IL2Ra_gc + kfbnd * IL2 * IL2Ra - k1rev * IL2_IL2Ra
    dydt[4] = -kfwd * IL2_IL2Rb * IL2Ra + k12rev * IL2_IL2Ra_IL2Rb - kfwd * IL2_IL2Rb * gc + k5rev * IL2_IL2Rb_gc + kfbnd * IL2 * IL2Rb - k2rev * IL2_IL2Rb
    dydt[5] = -kfwd *IL2_gc * IL2Ra + k6rev * IL2_IL2Ra_gc - kfwd * IL2_gc * IL2Rb + k7rev * IL2_IL2Rb_gc + k3fwd * IL2 * gc - k3rev * IL2_gc
    dydt[6] = -kfwd * IL2_IL2Ra_IL2Rb * gc + k10rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra * IL2Rb - k11rev * IL2_IL2Ra_IL2Rb + kfwd * IL2_IL2Rb * IL2Ra - k12rev * IL2_IL2Ra_IL2Rb
    dydt[7] = -kfwd * IL2_IL2Ra_gc * IL2Rb + k9rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra * gc - k4rev * IL2_IL2Ra_gc + kfwd * IL2_gc * IL2Ra - k6rev * IL2_IL2Ra_gc
    dydt[8] = -kfwd * IL2_IL2Rb_gc * IL2Ra + k8rev * IL2_IL2Ra_IL2Rb_gc + kfwd * gc * IL2_IL2Rb - k5rev * IL2_IL2Rb_gc + kfwd * IL2_gc * IL2Rb - k7rev * IL2_IL2Rb_gc
    dydt[9] = kfwd * IL2_IL2Rb_gc * IL2Ra - k8rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra_gc * IL2Rb - k9rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra_IL2Rb * gc - k10rev * IL2_IL2Ra_IL2Rb_gc

    # IL15
    dydt[10] = -kfbnd * IL15Ra * IL15 + k13rev * IL15_IL15Ra - kfbnd * IL15Ra * IL15_gc + k18rev * IL15_IL15Ra_gc - kfwd * IL15Ra * IL15_IL2Rb_gc + k20rev * IL15_IL15Ra_IL2Rb_gc - kfwd * IL15Ra * IL15_IL2Rb + k24rev * IL15_IL15Ra_IL2Rb
    dydt[11] = -kfwd * IL15_IL15Ra * IL2Rb + k23rev * IL15_IL15Ra_IL2Rb - kfwd * IL15_IL15Ra * gc + k16rev * IL15_IL15Ra_gc + kfbnd * IL15 * IL15Ra - k13rev * IL15_IL15Ra
    dydt[12] = -kfwd * IL15_IL2Rb * IL15Ra + k24rev * IL15_IL15Ra_IL2Rb - kfbnd * IL15_IL2Rb * gc + k17rev * IL15_IL2Rb_gc + kfbnd * IL15 * IL2Rb - k14rev * IL15_IL2Rb 
    dydt[13] = -kfbnd * IL15_gc * IL15Ra + k18rev * IL15_IL15Ra_gc - kfwd * IL15_gc * IL2Rb + k19rev * IL15_IL2Rb_gc + kfbnd * IL15 * gc - k15rev * IL15_gc
    dydt[14] = -kfwd * IL15_IL15Ra_IL2Rb * gc + k22rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra * IL2Rb - k23rev * IL15_IL15Ra_IL2Rb + kfwd * IL15_IL2Rb * IL15Ra - k24rev * IL15_IL15Ra_IL2Rb   
    dydt[15] = -kfwd * IL15_IL15Ra_gc * IL2Rb + k21rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra * gc - k16rev * IL15_IL15Ra_gc + kfbnd * IL15_gc * IL15Ra - k18rev * IL15_IL15Ra_gc
    dydt[16] = -kfwd * IL15_IL2Rb_gc * IL15Ra + k20rev * IL15_IL15Ra_IL2Rb_gc + kfbnd * gc * IL15_IL2Rb - k17rev * IL15_IL2Rb_gc + kfwd * IL15_gc * IL2Rb - k19rev * IL15_IL2Rb_gc
    dydt[17] =  kfwd * IL15_IL2Rb_gc * IL15Ra - k20rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra_gc * IL2Rb - k21rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra_IL2Rb * gc - k22rev * IL15_IL15Ra_IL2Rb_gc
    
    dydt[1] = dydt[1] - kfbnd * IL2Rb * IL15 + k14rev * IL15_IL2Rb - kfwd * IL2Rb * IL15_gc + k19rev * IL15_IL2Rb_gc - kfwd * IL2Rb * IL15_IL15Ra_gc + k21rev * IL15_IL15Ra_IL2Rb_gc - kfwd * IL2Rb * IL15_IL15Ra + k23rev * IL15_IL15Ra_IL2Rb
    dydt[2] = dydt[2] - kfbnd * IL15 * gc + k15rev * IL15_gc - kfbnd * IL15_IL2Rb * gc + k17rev * IL15_IL2Rb_gc - kfwd * IL15_IL15Ra * gc + k16rev * IL15_IL15Ra_gc - kfwd * IL15_IL15Ra_IL2Rb * gc + k22rev * IL15_IL15Ra_IL2Rb_gc
    
    # IL7
    dydt[2] = dydt[2] - kfbnd * IL7 * gc + k26rev * gc_IL7 - kfwd * gc * IL7Ra_IL7 + k27rev * IL7Ra_gc_IL7
    dydt[18] = -kfbnd * IL7Ra * IL7 + k25rev * IL7Ra_IL7 - kfwd * IL7Ra * gc_IL7 + k28rev * IL7Ra_gc_IL7
    dydt[19] = kfbnd * IL7Ra * IL7 - k25rev * IL7Ra_IL7 - kfwd * gc * IL7Ra_IL7 + k27rev * IL7Ra_gc_IL7
    dydt[20] = -kfwd * IL7Ra * gc_IL7 + k28rev * IL7Ra_gc_IL7 + kfbnd * IL7 * gc - k26rev * gc_IL7
    dydt[21] = kfwd * IL7Ra * gc_IL7 - k28rev * IL7Ra_gc_IL7 + kfwd * gc * IL7Ra_IL7 - k27rev * IL7Ra_gc_IL7

    # IL9
    dydt[2] = dydt[2] - kfbnd * IL9 * gc + k30rev * gc_IL9 - kfwd * gc * IL9R_IL9 + k31rev * IL9R_gc_IL9
    dydt[22] = -kfbnd * IL9R * IL9 + k29rev * IL9R_IL9 - kfwd * IL9R * gc_IL9 + k32rev * IL9R_gc_IL9
    dydt[23] = kfbnd * IL9R * IL9 - k29rev * IL9R_IL9 - kfwd * gc * IL9R_IL9 + k31rev * IL9R_gc_IL9
    dydt[24] = -kfwd * IL9R * gc_IL9 + k32rev * IL9R_gc_IL9 + kfbnd * IL9 * gc - k30rev * gc_IL9
    dydt[25] = kfwd * IL9R * gc_IL9 - k32rev * IL9R_gc_IL9 + kfwd * gc * IL9R_IL9 - k31rev * IL9R_gc_IL9

    return dydt


def subset_wrapper(y, t, IL2i=None, IL15i=None, IL9i=None, IL7i=None, **kwargs):
    ''' Wrapper function to only handle certain cytokines. '''

    IDX = np.zeros(26, dtype=np.bool)
    ys = np.zeros(26)
    kw = dict(kwargs)

    IDX[2] = 1 # Always need the gc

    if IL2i is None:
        kw['k5rev'] = kw['k6rev'] = 1.0
        kw['IL2'] = 0.0
    else:
        kw['IL2'] = IL2i
        IDX[0:10] = 1 # Set the first value in y to be IL2Ra

    if IL15i is None:
        kw['k15rev'] = kw['k17rev'] = kw['k18rev'] = kw['k22rev'] = kw['k23rev'] = 1.0
        kw['IL15'] = 0.0
    else:
        kw['IL15'] = IL15i
        IDX[10:18] = 1
        IDX[0] = 1 # Also need IL2Ra

    if IL7i is None:
        kw['k26rev'] = kw['k27rev'] = 1.0
        kw['IL7'] = 0.0
    else:
        kw['IL7'] = IL7i
        IDX[18:22] = 1

    if IL9i is None:
        kw['k29rev'] = kw['k30rev'] = kw['k31rev'] = 1.0
        kw['IL9'] = 0.0
    else:
        kw['IL9'] = IL9i
        IDX[22:26] = 1

    ys[IDX] = y
    ret_val = dy_dt(ys, t, **kw)

    return ret_val[IDX]
