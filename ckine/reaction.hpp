#ifndef RXN_CODE_ONCE_
#define RXN_CODE_ONCE_

std::array<bool, halfL> __active_species_IDX() {
	std::array<bool, halfL> __active_species_IDX;
	std::fill(__active_species_IDX.begin(), __active_species_IDX.end(), false);

	__active_species_IDX[7] = true;
	__active_species_IDX[8] = true;
	__active_species_IDX[14] = true;
	__active_species_IDX[15] = true;
	__active_species_IDX[18] = true;
	__active_species_IDX[21] = true;
	__active_species_IDX[24] = true;
	__active_species_IDX[27] = true;

	return __active_species_IDX;
}

const std::array<bool, halfL> activeV = __active_species_IDX();

const std::array<size_t, 8> recIDX = {{0, 1, 2, 9, 16, 19, 22, 25}};


/**
 * @brief      k12rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k12rev.
 */
double k12rev(const bindingRates * const r) {
	return r->k1rev * r->k11rev / r->k2rev; // loop for IL2_IL2Ra_IL2Rb
}


/**
 * @brief      k9rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k9rev.
 */
double k9rev(const bindingRates * const r) {
	return r->k10rev * r->k11rev / r->k4rev; // Based on formation of full complex (IL2_IL2Ra_IL2Rb_gc)
}


/**
 * @brief      k8rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k8rev.
 */
double k8rev(const bindingRates * const r) {
	return r->k10rev * k12rev(r) / r->k5rev; // Based on formation of full complex (IL2_IL2Ra_IL2Rb_gc)
}


/**
 * @brief      k24rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k24rev.
 */
double k24rev(const bindingRates * const r) {
	return k13rev * r->k23rev / k14rev;
}


/**
 * @brief      k21rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k21rev.
 */
double k21rev(const bindingRates * const r) {
	return r->k22rev * r->k23rev / r->k16rev;
}


/**
 * @brief      k20rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k20rev.
 */
double k20rev(const bindingRates * const r) {
	return r->k22rev * k24rev(r) / r->k17rev;
}


void dy_dt(const double * const y, const bindingRates * const r, double * const dydt, const double * const ILs) {
	// IL2 in nM
	const double IL2Ra = y[0];
	const double IL2Rb = y[1];
	const double gc = y[2];
	const double IL2_IL2Ra = y[3];
	const double IL2_IL2Rb = y[4];
	const double IL2_IL2Ra_IL2Rb = y[5];
	const double IL2_IL2Ra_gc = y[6];
	const double IL2_IL2Rb_gc = y[7];
	const double IL2_IL2Ra_IL2Rb_gc = y[8];
	
	// IL15 in nM
	const double IL15Ra = y[9];
	const double IL15_IL15Ra = y[10];
	const double IL15_IL2Rb = y[11];
	const double IL15_IL15Ra_IL2Rb = y[12];
	const double IL15_IL15Ra_gc = y[13];
	const double IL15_IL2Rb_gc = y[14];
	const double IL15_IL15Ra_IL2Rb_gc = y[15];
		
	// IL2
	dydt[0] = -kfbnd * IL2Ra * ILs[0] + r->k1rev * IL2_IL2Ra - r->kfwd * IL2Ra * IL2_IL2Rb_gc + k8rev(r) * IL2_IL2Ra_IL2Rb_gc - r->kfwd * IL2Ra * IL2_IL2Rb + k12rev(r) * IL2_IL2Ra_IL2Rb;
	dydt[1] = -kfbnd * IL2Rb * ILs[0] + r->k2rev * IL2_IL2Rb - r->kfwd * IL2Rb * IL2_IL2Ra_gc + k9rev(r) * IL2_IL2Ra_IL2Rb_gc - r->kfwd * IL2Rb * IL2_IL2Ra + r->k11rev * IL2_IL2Ra_IL2Rb;
	dydt[2] = -r->kfwd * IL2_IL2Rb * gc + r->k5rev * IL2_IL2Rb_gc - r->kfwd * IL2_IL2Ra * gc + r->k4rev * IL2_IL2Ra_gc - r->kfwd * IL2_IL2Ra_IL2Rb * gc + r->k10rev * IL2_IL2Ra_IL2Rb_gc;
	dydt[3] = -r->kfwd * IL2_IL2Ra * IL2Rb + r->k11rev * IL2_IL2Ra_IL2Rb - r->kfwd * IL2_IL2Ra * gc + r->k4rev * IL2_IL2Ra_gc + kfbnd * ILs[0] * IL2Ra - r->k1rev * IL2_IL2Ra;
	dydt[4] = -r->kfwd * IL2_IL2Rb * IL2Ra + k12rev(r) * IL2_IL2Ra_IL2Rb - r->kfwd * IL2_IL2Rb * gc + r->k5rev * IL2_IL2Rb_gc + kfbnd * ILs[0] * IL2Rb - r->k2rev * IL2_IL2Rb;
	dydt[5] = -r->kfwd * IL2_IL2Ra_IL2Rb * gc + r->k10rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra * IL2Rb - r->k11rev * IL2_IL2Ra_IL2Rb + r->kfwd * IL2_IL2Rb * IL2Ra - k12rev(r) * IL2_IL2Ra_IL2Rb;
	dydt[6] = -r->kfwd * IL2_IL2Ra_gc * IL2Rb + k9rev(r) * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra * gc - r->k4rev * IL2_IL2Ra_gc;
	dydt[7] = -r->kfwd * IL2_IL2Rb_gc * IL2Ra + k8rev(r) * IL2_IL2Ra_IL2Rb_gc + r->kfwd * gc * IL2_IL2Rb - r->k5rev * IL2_IL2Rb_gc;
	dydt[8] = r->kfwd * IL2_IL2Rb_gc * IL2Ra - k8rev(r) * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra_gc * IL2Rb - k9rev(r) * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra_IL2Rb * gc - r->k10rev * IL2_IL2Ra_IL2Rb_gc;

	// IL15
	dydt[9] = -kfbnd * IL15Ra * ILs[1] + k13rev * IL15_IL15Ra - r->kfwd * IL15Ra * IL15_IL2Rb_gc + k20rev(r) * IL15_IL15Ra_IL2Rb_gc - r->kfwd * IL15Ra * IL15_IL2Rb + k24rev(r) * IL15_IL15Ra_IL2Rb;
	dydt[10] = -r->kfwd * IL15_IL15Ra * IL2Rb + r->k23rev * IL15_IL15Ra_IL2Rb - r->kfwd * IL15_IL15Ra * gc + r->k16rev * IL15_IL15Ra_gc + kfbnd * ILs[1] * IL15Ra - k13rev * IL15_IL15Ra;
	dydt[11] = -r->kfwd * IL15_IL2Rb * IL15Ra + k24rev(r) * IL15_IL15Ra_IL2Rb - r->kfwd * IL15_IL2Rb * gc + r->k17rev * IL15_IL2Rb_gc + kfbnd * ILs[1] * IL2Rb - k14rev * IL15_IL2Rb;
	dydt[12] = -r->kfwd * IL15_IL15Ra_IL2Rb * gc + r->k22rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra * IL2Rb - r->k23rev * IL15_IL15Ra_IL2Rb + r->kfwd * IL15_IL2Rb * IL15Ra - k24rev(r) * IL15_IL15Ra_IL2Rb;
	dydt[13] = -r->kfwd * IL15_IL15Ra_gc * IL2Rb + k21rev(r) * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra * gc - r->k16rev * IL15_IL15Ra_gc;
	dydt[14] = -r->kfwd * IL15_IL2Rb_gc * IL15Ra + k20rev(r) * IL15_IL15Ra_IL2Rb_gc + r->kfwd * gc * IL15_IL2Rb - r->k17rev * IL15_IL2Rb_gc;
	dydt[15] =  r->kfwd * IL15_IL2Rb_gc * IL15Ra - k20rev(r) * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra_gc * IL2Rb - k21rev(r) * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra_IL2Rb * gc - r->k22rev * IL15_IL15Ra_IL2Rb_gc;
	
	dydt[1] = dydt[1] - kfbnd * IL2Rb * ILs[1] + k14rev * IL15_IL2Rb - r->kfwd * IL2Rb * IL15_IL15Ra_gc + k21rev(r) * IL15_IL15Ra_IL2Rb_gc - r->kfwd * IL2Rb * IL15_IL15Ra + r->k23rev * IL15_IL15Ra_IL2Rb;
	dydt[2] = dydt[2] - r->kfwd * IL15_IL2Rb * gc + r->k17rev * IL15_IL2Rb_gc - r->kfwd * IL15_IL15Ra * gc + r->k16rev * IL15_IL15Ra_gc - r->kfwd * IL15_IL15Ra_IL2Rb * gc + r->k22rev * IL15_IL15Ra_IL2Rb_gc; 
	
	auto simpleCkine = [&](const size_t ij, const double revOne, const double revTwo, const double IL) {
		dydt[2] += - r->kfwd * gc * y[ij+1] + revTwo * y[ij+2];
		dydt[ij] = -kfbnd * y[ij] * IL + revOne * y[ij+1];
		dydt[ij+1] = kfbnd * y[ij] * IL - revOne * y[ij+1] - r->kfwd * gc * y[ij+1] + revTwo * y[ij+2];
		dydt[ij+2] = r->kfwd * gc * y[ij+1] - revTwo * y[ij+2];
	};

	simpleCkine(16, k25rev, r->k27rev, ILs[2]);
	simpleCkine(19, k29rev, r->k31rev, ILs[3]);
	simpleCkine(22, k32rev, r->k33rev, ILs[4]);
	simpleCkine(25, k34rev, r->k35rev, ILs[5]);
}


void fullModel(const double * const y, const ratesS * const r, double *dydt) {
	// Implement full model.
	std::fill(dydt, dydt + Nspecies, 0.0);

	// Calculate cell surface and endosomal reactions
	dy_dt(y,          &r->surface,         dydt, r->ILs.data());
	dy_dt(y + halfL, &r->endosome, dydt + halfL,   y + halfL*2);

	// Handle endosomal ligand balance.
	// Must come before trafficking as we only calculate this based on reactions balance
	double const * const dydti = dydt + halfL;
	dydt[56] = -std::accumulate(dydti+3,  dydti+9, (double) 0.0) / internalV;
	dydt[57] = -std::accumulate(dydti+10, dydti+16, (double) 0.0) / internalV;
	dydt[58] = -std::accumulate(dydti+17, dydti+19, (double) 0.0) / internalV;
	dydt[59] = -std::accumulate(dydti+20, dydti+22, (double) 0.0) / internalV;
	dydt[60] = -std::accumulate(dydti+23, dydti+25, (double) 0.0) / internalV;
	dydt[61] = -std::accumulate(dydti+26, dydti+28, (double) 0.0) / internalV;

	// Actually calculate the trafficking
	for (size_t ii = 0; ii < halfL; ii++) {
		if (activeV[ii]) {
			dydt[ii] += -y[ii]*(r->endo + r->activeEndo); // Endocytosis
			dydt[ii+halfL] += y[ii]*(r->endo + r->activeEndo)/internalFrac - r->kDeg*y[ii+halfL]; // Endocytosis, degradation
		} else {
			dydt[ii] += -y[ii]*r->endo + r->kRec*(1.0-r->sortF)*y[ii+halfL]*internalFrac; // Endocytosis, recycling
			dydt[ii+halfL] += y[ii]*r->endo/internalFrac - r->kRec*(1.0-r->sortF)*y[ii+halfL] - (r->kDeg*r->sortF)*y[ii+halfL]; // Endocytosis, recycling, degradation
		}
	}

	// Expression: IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R, IL4Ra, IL21Ra
	for (size_t ii = 0; ii < recIDX.size(); ii++)
		dydt[recIDX[ii]] += r->Rexpr[ii];

	// Degradation does lead to some clearance of ligand in the endosome
	for (size_t ii = 0; ii < 6; ii++)
		dydt[(halfL*2) + ii] -= y[(halfL*2) + ii] * r->kDeg;
}


std::array<double, Nspecies> solveAutocrine(const ratesS * const r) {
	std::array<double, Nspecies> y0;
	std::fill(y0.begin(), y0.end(), 0.0);

	// Expand out trafficking terms
	const double kRec = r->kRec*(1-r->sortF);
	const double kDeg = r->kDeg*r->sortF;

	// Assuming no autocrine ligand, so can solve steady state
	// Add the species
	for (size_t ii = 0; ii < recIDX.size(); ii++) {
		y0[recIDX[ii] + halfL] = r->Rexpr[ii] / kDeg / internalFrac;
		y0[recIDX[ii]] = (r->Rexpr[ii] + kRec*y0[recIDX[ii] + halfL]*internalFrac)/r->endo;
	}

	return y0;
}


/**
 * @brief      Setup the autocrine state sensitivities.
 *
 * @param[in]  r     Rate parameters.
 * @param      y0s   The autocrine state sensitivities.
 */
void solveAutocrineS (const ratesS * const r, N_Vector *y0s) {
	std::array<double, Nspecies> y0 = solveAutocrine(r);

	for (size_t is = 0; is < Nparams; is++)
		N_VConst(0.0, y0s[is]);

	for (size_t is : recIDX) {
		// Endosomal amount doesn't depend on endo
		NV_Ith_S(y0s[17], is) = -y0[is]/r->endo; // Endo (17)

		// sortF (19)
		NV_Ith_S(y0s[19], is + halfL) = -y0[is + halfL]/r->sortF;
		NV_Ith_S(y0s[19], is) = r->kRec*internalFrac/r->endo*((1 - r->sortF)*NV_Ith_S(y0s[19], is + halfL) - y0[is + halfL]);

		// Endosomal amount doesn't depend on kRec
		NV_Ith_S(y0s[20], is) = (1-r->sortF)*y0[is + halfL]*internalFrac/r->endo; // kRec (20)

		// kDeg (21)
		NV_Ith_S(y0s[21], is + halfL) = -y0[is + halfL]/r->kDeg;
		NV_Ith_S(y0s[21], is) = r->kRec*(1-r->sortF)*NV_Ith_S(y0s[21], is + halfL)*internalFrac/r->endo;
	}

	// Rexpr (22-30)
	for (size_t ii = 0; ii < recIDX.size(); ii++) {
		NV_Ith_S(y0s[22 + ii], recIDX[ii] + halfL) = y0[recIDX[ii] + halfL]/r->Rexpr[ii];
		NV_Ith_S(y0s[22 + ii], recIDX[ii]) = 1/r->endo + NV_Ith_S(y0s[22 + ii], recIDX[ii] + halfL)*r->kRec*(1-r->sortF)*internalFrac/r->endo;
	}
}

#endif
