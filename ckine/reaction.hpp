#ifndef RXN_CODE_ONCE_
#define RXN_CODE_ONCE_

#include <nvector/nvector_serial.h>

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
template <class T>
T k12rev(const bindingRates<T> * const r) {
	return r->k1rev * r->k11rev / r->k2rev; // loop for IL2_IL2Ra_IL2Rb
}


/**
 * @brief      k9rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k9rev.
 */
template <class T>
T k9rev(const bindingRates<T> * const r) {
	return r->k10rev * r->k11rev / r->k4rev; // Based on formation of full complex (IL2_IL2Ra_IL2Rb_gc)
}


/**
 * @brief      k8rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k8rev.
 */
template <class T>
T k8rev(const bindingRates<T> * const r) {
	return r->k10rev * k12rev(r) / r->k5rev; // Based on formation of full complex (IL2_IL2Ra_IL2Rb_gc)
}


/**
 * @brief      k24rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k24rev.
 */
template <class T>
T k24rev(const bindingRates<T> * const r) {
	return r->k13rev * r->k23rev / r->k14rev;
}


/**
 * @brief      k21rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k21rev.
 */
template <class T>
T k21rev(const bindingRates<T> * const r) {
	return r->k22rev * r->k23rev / r->k16rev;
}


/**
 * @brief      k20rev value. Must hold to satisfy detailed balance.
 *
 * @param[in]  r     bindingRates structure.
 *
 * @return     Value of k20rev.
 */
template <class T>
T k20rev(const bindingRates<T> * const r) {
	return r->k22rev * k24rev(r) / r->k17rev;
}


template <class T, class YT, class RT, class LT>
void dy_dt(const YT * const y, bindingRates<RT> * r, T *dydt, LT *ILs) {
	// IL2 in nM
	const YT IL2Ra = y[0];
	const YT IL2Rb = y[1];
	const YT gc = y[2];
	const YT IL2_IL2Ra = y[3];
	const YT IL2_IL2Rb = y[4];
	const YT IL2_IL2Ra_IL2Rb = y[5];
	const YT IL2_IL2Ra_gc = y[6];
	const YT IL2_IL2Rb_gc = y[7];
	const YT IL2_IL2Ra_IL2Rb_gc = y[8];
	
	// IL15 in nM
	const YT IL15Ra = y[9];
	const YT IL15_IL15Ra = y[10];
	const YT IL15_IL2Rb = y[11];
	const YT IL15_IL15Ra_IL2Rb = y[12];
	const YT IL15_IL15Ra_gc = y[13];
	const YT IL15_IL2Rb_gc = y[14];
	const YT IL15_IL15Ra_IL2Rb_gc = y[15];
		
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
	dydt[9] = -kfbnd * IL15Ra * ILs[1] + r->k13rev * IL15_IL15Ra - r->kfwd * IL15Ra * IL15_IL2Rb_gc + k20rev(r) * IL15_IL15Ra_IL2Rb_gc - r->kfwd * IL15Ra * IL15_IL2Rb + k24rev(r) * IL15_IL15Ra_IL2Rb;
	dydt[10] = -r->kfwd * IL15_IL15Ra * IL2Rb + r->k23rev * IL15_IL15Ra_IL2Rb - r->kfwd * IL15_IL15Ra * gc + r->k16rev * IL15_IL15Ra_gc + kfbnd * ILs[1] * IL15Ra - r->k13rev * IL15_IL15Ra;
	dydt[11] = -r->kfwd * IL15_IL2Rb * IL15Ra + k24rev(r) * IL15_IL15Ra_IL2Rb - r->kfwd * IL15_IL2Rb * gc + r->k17rev * IL15_IL2Rb_gc + kfbnd * ILs[1] * IL2Rb - r->k14rev * IL15_IL2Rb;
	dydt[12] = -r->kfwd * IL15_IL15Ra_IL2Rb * gc + r->k22rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra * IL2Rb - r->k23rev * IL15_IL15Ra_IL2Rb + r->kfwd * IL15_IL2Rb * IL15Ra - k24rev(r) * IL15_IL15Ra_IL2Rb;
	dydt[13] = -r->kfwd * IL15_IL15Ra_gc * IL2Rb + k21rev(r) * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra * gc - r->k16rev * IL15_IL15Ra_gc;
	dydt[14] = -r->kfwd * IL15_IL2Rb_gc * IL15Ra + k20rev(r) * IL15_IL15Ra_IL2Rb_gc + r->kfwd * gc * IL15_IL2Rb - r->k17rev * IL15_IL2Rb_gc;
	dydt[15] =  r->kfwd * IL15_IL2Rb_gc * IL15Ra - k20rev(r) * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra_gc * IL2Rb - k21rev(r) * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra_IL2Rb * gc - r->k22rev * IL15_IL15Ra_IL2Rb_gc;
	
	dydt[1] = dydt[1] - kfbnd * IL2Rb * ILs[1] + r->k14rev * IL15_IL2Rb - r->kfwd * IL2Rb * IL15_IL15Ra_gc + k21rev(r) * IL15_IL15Ra_IL2Rb_gc - r->kfwd * IL2Rb * IL15_IL15Ra + r->k23rev * IL15_IL15Ra_IL2Rb;
	dydt[2] = dydt[2] - r->kfwd * IL15_IL2Rb * gc + r->k17rev * IL15_IL2Rb_gc - r->kfwd * IL15_IL15Ra * gc + r->k16rev * IL15_IL15Ra_gc - r->kfwd * IL15_IL15Ra_IL2Rb * gc + r->k22rev * IL15_IL15Ra_IL2Rb_gc; 

	auto simpleCkine = [&](const size_t ij, const T revOne, const T revTwo, const LT IL) {
		dydt[2] += - r->kfwd * gc * y[ij+1] + revTwo * y[ij+2];
		dydt[ij] = -kfbnd * y[ij] * IL + revOne * y[ij+1];
		dydt[ij+1] = kfbnd * y[ij] * IL - revOne * y[ij+1] - r->kfwd * gc * y[ij+1] + revTwo * y[ij+2];
		dydt[ij+2] = r->kfwd * gc * y[ij+1] - revTwo * y[ij+2];
	};

	simpleCkine(16, r->k25rev, r->k27rev, ILs[2]);
	simpleCkine(19, r->k29rev, r->k31rev, ILs[3]);
	simpleCkine(22, r->k32rev, r->k33rev, ILs[4]);
	simpleCkine(25, r->k34rev, r->k35rev, ILs[5]);
}


template <class T, class YT, class RT>
void fullModel(const YT * const y, ratesS<RT> * const r, T *dydt) {
	// Implement full model.
	std::fill(dydt, dydt + Nspecies, 0.0);

	// Calculate cell surface and endosomal reactions
	dy_dt(y,          &r->surface,         dydt, r->ILs.data());
	dy_dt(y + halfL, &r->endosome, dydt + halfL,   y + halfL*2);

	// Handle endosomal ligand balance.
	// Must come before trafficking as we only calculate this based on reactions balance
	T *dydti = dydt + halfL;
	dydt[56] = -std::accumulate(dydti+3,  dydti+9, (T) 0.0) / internalV;
	dydt[57] = -std::accumulate(dydti+10, dydti+16, (T) 0.0) / internalV;
	dydt[58] = -std::accumulate(dydti+17, dydti+19, (T) 0.0) / internalV;
	dydt[59] = -std::accumulate(dydti+20, dydti+22, (T) 0.0) / internalV;
	dydt[60] = -std::accumulate(dydti+23, dydti+25, (T) 0.0) / internalV;
	dydt[61] = -std::accumulate(dydti+26, dydti+28, (T) 0.0) / internalV;

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


template <class T>
std::array<T, Nspecies> solveAutocrine(const ratesS<T> * const r) {
	std::array<T, Nspecies> y0;
	std::fill(y0.begin(), y0.end(), 0.0);

	// Check if we're working with the no trafficking model
	if (r->endo == 0.0) {
		for (size_t ii = 0; ii < recIDX.size(); ii++) {
			y0[recIDX[ii]] = r->Rexpr[ii];
		}

		return y0;
	}

	// Expand out trafficking terms
	const T kRec = r->kRec*(1-r->sortF);
	const T kDeg = r->kDeg*r->sortF;

	// Assuming no autocrine ligand, so can solve steady state
	// Add the species
	for (size_t ii = 0; ii < recIDX.size(); ii++) {
		y0[recIDX[ii] + halfL] = r->Rexpr[ii] / kDeg / internalFrac;
		y0[recIDX[ii]] = (r->Rexpr[ii] + kRec*y0[recIDX[ii] + halfL]*internalFrac)/r->endo;
	}

	return y0;
}

#endif
