import numpy as np



def dy_dt(y, t, IL2, k1fwd, k4fwd, k4rev, k5rev, k8rev, k11rev):
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

	# The receptor-receptor forward rate is largely going to be determined by plasma membrane diffusion
	# so we're going to assume it's shared.
	k5fwd = k4fwd
	k6fwd = k4fwd
	k7fwd = k4fwd
	k8fwd = k4fwd
	k9fwd = k4fwd
	k10fwd = k4fwd
	k11fwd = k4fwd
	k12fwd = k4fwd

	# To satisfy detailed balance these relationships should hold
	# TODO: Will be specified later
	k10rev = k8rev * 0 * 0
	k9rev = k8rev * 0 * 0

	k6rev = k4rev * 0 * 0
	k7rev = k5rev * 0 * 0
	k12rev = k11rev * 0 * 0

	# These are measured in the literature
	k1rev = 0
	k2fwd = 0
	k2rev = 0
	k3fwd = 0
	k3rev = 0


	dydt = np.zeros(y.shape, dtype = np.float64)

	dydt[0] = -k1fwd * IL2Ra * IL2 + k1rev * IL2_IL2Ra - k6fwd * IL2Ra * IL2_gc - k6rev * IL2_IL2Ra_gc - k8fwd * IL2Ra * IL2_IL2Rb_gc + k8rev * IL2_IL2Ra_IL2Rb_gc - k12fwd * IL2Ra * IL2_IL2Rb + k12rev * IL2_IL2Ra_IL2Rb
	dydt[1] = -k2fwd * IL2Rb * IL2 + k2rev * IL2_IL2Rb - k7fwd * IL2Rb * IL2_gc + k7rev * IL2_IL2Rb_gc - k9fwd * IL2Rb * IL2_IL2Ra_gc + k9rev * IL2_IL2Ra_IL2Rb_gc - k11fwd * IL2Rb * IL2_IL2Ra + k11rev * IL2_IL2Ra_IL2Rb
	dydt[2] = -k3fwd * IL2 * gc + k3rev * IL2_gc - k5fwd * IL2_IL2Rb * gc + k5rev * IL2_IL2Rb_gc - k4fwd * IL2_IL2Ra * gc + k4rev * IL2_IL2Ra_gc - k10fwd * IL2_IL2Ra_IL2Rb * gc + k10rev * IL2_IL2Ra_IL2Rb_gc
	dydt[3] = -k11fwd * IL2_IL2Ra * IL2Rb + k11rev * IL2_IL2Ra_IL2Rb - k4fwd * IL2_IL2Ra * gc + k4rev * IL2_IL2Ra_gc + k1fwd * IL2 * IL2Ra - k1rev * IL2_IL2Ra
	dydt[4] = -k12fwd * IL2_IL2Rb * IL2Ra + k12rev * IL2_IL2Ra_IL2Rb - k5fwd * IL2_IL2Rb * gc + k5rev * IL2_IL2Rb_gc + k2fwd * IL2 * IL2Rb - k2rev * IL2_IL2Rb
	dydt[5] = -k6fwd *IL2_gc * IL2Ra + k6rev * IL2_IL2Ra_gc - k7fwd * IL2_gc * IL2Rb + k7rev * IL2_IL2Rb_gc + k3fwd * IL2 * gc - k3rev * IL2_gc
	dydt[6] = -k10fwd * IL2_IL2Ra_IL2Rb * gc + k10rev * IL2_IL2Ra_IL2Rb_gc + k11fwd * IL2_IL2Ra * IL2Rb - k11rev * IL2_IL2Ra_IL2Rb + k12fwd * IL2_IL2Rb * IL2Ra - k12rev * IL2_IL2Ra_IL2Rb
	dydt[7] = -k9fwd * IL2_IL2Ra_gc * IL2Rb + k9rev * IL2_IL2Ra_IL2Rb_gc + k4fwd * IL2_IL2Ra * gc - k4rev * IL2_IL2Ra_gc + k6fwd * IL2_gc * IL2Ra - k6rev * IL2_IL2Ra_gc
	dydt[8] = -k8fwd * IL2_IL2Rb_gc * IL2Ra + k8rev * IL2_IL2Ra_IL2Rb_gc + k5fwd * gc * IL2_IL2Rb - k5rev * IL2_IL2Rb_gc + k7fwd * IL2_gc * IL2Rb - k7rev * IL2_IL2Rb_gc
	dydt[9] = k8fwd * IL2_IL2Rb_gc * IL2Ra - k8rev * IL2_IL2Ra_IL2Rb_gc + k9fwd * IL2_IL2Ra_gc * IL2Rb - k9rev * IL2_IL2Ra_IL2Rb_gc + k10fwd * IL2_IL2Ra_IL2Rb * gc - k10rev * IL2_IL2Ra_IL2Rb_gc
	
	# added dydt[2] through dydt[9] based on the diagram pictured in type-I-ckine-model/model/graph.pdf on 9/19/17 by Adam; dydt[0] and dydt[1] were done by Aaron

	return dydt