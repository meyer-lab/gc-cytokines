
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


void dy_dt(const double * const y, const ratesS * const r, double * const dydt, const double * const ILs) {
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
	dydt[0] = -kfbnd * IL2Ra * ILs[0] + k1rev * IL2_IL2Ra - r->kfwd * IL2Ra * IL2_IL2Rb_gc + r->k8rev * IL2_IL2Ra_IL2Rb_gc - r->kfwd * IL2Ra * IL2_IL2Rb + r->k12rev * IL2_IL2Ra_IL2Rb;
	dydt[1] = -kfbnd * IL2Rb * ILs[0] + k2rev * IL2_IL2Rb - r->kfwd * IL2Rb * IL2_IL2Ra_gc + r->k9rev * IL2_IL2Ra_IL2Rb_gc - r->kfwd * IL2Rb * IL2_IL2Ra + r->k11rev * IL2_IL2Ra_IL2Rb;
	dydt[2] = -r->kfwd * IL2_IL2Rb * gc + r->k5rev * IL2_IL2Rb_gc - r->kfwd * IL2_IL2Ra * gc + r->k4rev * IL2_IL2Ra_gc - r->kfwd * IL2_IL2Ra_IL2Rb * gc + r->k10rev * IL2_IL2Ra_IL2Rb_gc;
	dydt[3] = -r->kfwd * IL2_IL2Ra * IL2Rb + r->k11rev * IL2_IL2Ra_IL2Rb - r->kfwd * IL2_IL2Ra * gc + r->k4rev * IL2_IL2Ra_gc + kfbnd * ILs[0] * IL2Ra - k1rev * IL2_IL2Ra;
	dydt[4] = -r->kfwd * IL2_IL2Rb * IL2Ra + r->k12rev * IL2_IL2Ra_IL2Rb - r->kfwd * IL2_IL2Rb * gc + r->k5rev * IL2_IL2Rb_gc + kfbnd * ILs[0] * IL2Rb - k2rev * IL2_IL2Rb;
	dydt[5] = -r->kfwd * IL2_IL2Ra_IL2Rb * gc + r->k10rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra * IL2Rb - r->k11rev * IL2_IL2Ra_IL2Rb + r->kfwd * IL2_IL2Rb * IL2Ra - r->k12rev * IL2_IL2Ra_IL2Rb;
	dydt[6] = -r->kfwd * IL2_IL2Ra_gc * IL2Rb + r->k9rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra * gc - r->k4rev * IL2_IL2Ra_gc;
	dydt[7] = -r->kfwd * IL2_IL2Rb_gc * IL2Ra + r->k8rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * gc * IL2_IL2Rb - r->k5rev * IL2_IL2Rb_gc;
	dydt[8] = r->kfwd * IL2_IL2Rb_gc * IL2Ra - r->k8rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra_gc * IL2Rb - r->k9rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra_IL2Rb * gc - r->k10rev * IL2_IL2Ra_IL2Rb_gc;

	// IL15
	dydt[9] = -kfbnd * IL15Ra * ILs[1] + k13rev * IL15_IL15Ra - r->kfwd * IL15Ra * IL15_IL2Rb_gc + r->k20rev * IL15_IL15Ra_IL2Rb_gc - r->kfwd * IL15Ra * IL15_IL2Rb + r->k24rev * IL15_IL15Ra_IL2Rb;
	dydt[10] = -r->kfwd * IL15_IL15Ra * IL2Rb + r->k23rev * IL15_IL15Ra_IL2Rb - r->kfwd * IL15_IL15Ra * gc + r->k16rev * IL15_IL15Ra_gc + kfbnd * ILs[1] * IL15Ra - k13rev * IL15_IL15Ra;
	dydt[11] = -r->kfwd * IL15_IL2Rb * IL15Ra + r->k24rev * IL15_IL15Ra_IL2Rb - r->kfwd * IL15_IL2Rb * gc + r->k17rev * IL15_IL2Rb_gc + kfbnd * ILs[1] * IL2Rb - k14rev * IL15_IL2Rb;
	dydt[12] = -r->kfwd * IL15_IL15Ra_IL2Rb * gc + r->k22rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra * IL2Rb - r->k23rev * IL15_IL15Ra_IL2Rb + r->kfwd * IL15_IL2Rb * IL15Ra - r->k24rev * IL15_IL15Ra_IL2Rb;
	dydt[13] = -r->kfwd * IL15_IL15Ra_gc * IL2Rb + r->k21rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra * gc - r->k16rev * IL15_IL15Ra_gc;
	dydt[14] = -r->kfwd * IL15_IL2Rb_gc * IL15Ra + r->k20rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * gc * IL15_IL2Rb - r->k17rev * IL15_IL2Rb_gc;
	dydt[15] =  r->kfwd * IL15_IL2Rb_gc * IL15Ra - r->k20rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra_gc * IL2Rb - r->k21rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra_IL2Rb * gc - r->k22rev * IL15_IL15Ra_IL2Rb_gc;
	
	dydt[1] = dydt[1] - kfbnd * IL2Rb * ILs[1] + k14rev * IL15_IL2Rb - r->kfwd * IL2Rb * IL15_IL15Ra_gc + r->k21rev * IL15_IL15Ra_IL2Rb_gc - r->kfwd * IL2Rb * IL15_IL15Ra + r->k23rev * IL15_IL15Ra_IL2Rb;
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
	dy_dt(y,         r,         dydt, r->ILs.data());
	dy_dt(y + halfL, r, dydt + halfL,   y + halfL*2);

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
