#include <algorithm>
#include <cstdio>
#include <numeric>
#include <array>
#include <vector>
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode.h>            /* prototypes for CVODE fcts., consts. */
#include <string>
#include <sundials/sundials_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <cvode/cvode_direct.h>
#include <iostream>
#include "model.hpp"

using std::array;
using std::copy;
using std::copy_n;
using std::vector;
using std::fill;
using std::string;

ratesS param(const double * const rxn, const double * const tfR) {
	ratesS r;

	r.IL2 = rxn[0];
	r.IL15 = rxn[1];
	r.IL7 = rxn[2];
	r.IL9 = rxn[3];
	r.kfwd = rxn[4];
	r.k5rev = rxn[5];
	r.k6rev = rxn[6];
	r.k15rev = rxn[7];
	r.k17rev = rxn[8];
	r.k18rev = rxn[9];
	r.k22rev = rxn[10];
	r.k23rev = rxn[11];
	r.k26rev = rxn[12];
	r.k27rev = rxn[13];
	r.k29rev = rxn[14];
	r.k30rev = rxn[15];
	r.k31rev = rxn[16];

	// These are probably measured in the literature
	r.kfbnd = 0.01; // Assuming on rate of 10^7 M-1 sec-1
	r.k1rev = r.kfbnd * 10; // doi:10.1016/j.jmb.2004.04.038, 10 nM

	r.k2rev = r.kfbnd * 144; // doi:10.1016/j.jmb.2004.04.038, 144 nM
	r.k3fwd = r.kfbnd / 10.0; // Very weak, > 50 uM. Voss, et al (1993). PNAS. 90, 2428–2432.
	r.k3rev = 50000 * r.k3fwd;
	r.k10rev = 12.0 * r.k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
	r.k11rev = 63.0 * r.k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
	
	// Literature values for k values for IL-15
	r.k13rev = r.kfbnd * 0.065; // based on the multiple papers suggesting 30-100 pM
	r.k14rev = r.kfbnd * 438; // doi:10.1038/ni.2449, 438 nM
	
	// Literature values for IL-7
	r.k25rev = r.kfbnd * 59; // DOI:10.1111/j.1600-065X.2012.01160.x, 59 nM
	
	// To satisfy detailed balance these relationships should hold
	// _Based on initial assembly steps
	r.k4rev = r.kfbnd * r.k6rev * r.k3rev / r.k1rev / r.k3fwd;
	r.k7rev = r.k3fwd * r.k2rev * r.k5rev / r.kfbnd / r.k3rev;
	r.k12rev = r.k1rev * r.k11rev / r.k2rev;
	// _Based on formation of full complex
	r.k9rev = r.k2rev * r.k10rev * r.k12rev / r.kfbnd / r.k3rev / r.k6rev * r.k3fwd;
	r.k8rev = r.k2rev * r.k10rev * r.k12rev / r.kfbnd / r.k7rev / r.k3rev * r.k3fwd;

	// IL15
	// To satisfy detailed balance these relationships should hold
	// _Based on initial assembly steps
	r.k16rev = r.kfwd * r.k18rev * r.k15rev / r.k13rev / r.kfbnd;
	r.k19rev = r.kfwd * r.k14rev * r.k17rev / r.kfbnd / r.k15rev;
	r.k24rev = r.k13rev * r.k23rev / r.k14rev;
	// _Based on formation of full complex

	r.k21rev = r.k14rev * r.k22rev * r.k24rev / r.kfwd / r.k15rev / r.k18rev * r.kfbnd;
	r.k20rev = r.k14rev * r.k22rev * r.k24rev / r.k19rev / r.k15rev;

	// _One detailed balance IL7/9 loop
	r.k32rev = r.k29rev * r.k31rev / r.k30rev;
	r.k28rev = r.k25rev * r.k27rev / r.k26rev;

	// Set the rates
	r.endo = tfR[0];
	r.activeEndo = tfR[1];
	r.sortF = tfR[2];
	r.kRec = tfR[3];
	r.kDeg = tfR[4];

	copy_n(tfR + 5, 6, r.Rexpr.begin());

	return r;
}

void dy_dt(const double * const y, const ratesS * const r, double * const dydt, double IL2, double IL15, double IL7, double IL9) {
	// IL2 in nM
	double IL2Ra = y[0];
	double IL2Rb = y[1];
	double gc = y[2];
	double IL2_IL2Ra = y[3];
	double IL2_IL2Rb = y[4];
	double IL2_gc = y[5];
	double IL2_IL2Ra_IL2Rb = y[6];
	double IL2_IL2Ra_gc = y[7];
	double IL2_IL2Rb_gc = y[8];
	double IL2_IL2Ra_IL2Rb_gc = y[9];
	
	// IL15 in nM
	double IL15Ra = y[10];
	double IL15_IL15Ra = y[11];
	double IL15_IL2Rb = y[12];
	double IL15_gc = y[13];
	double IL15_IL15Ra_IL2Rb = y[14];
	double IL15_IL15Ra_gc = y[15];
	double IL15_IL2Rb_gc = y[16];
	double IL15_IL15Ra_IL2Rb_gc = y[17];
	
	// IL7, IL9 in nM
	double IL7Ra = y[18];
	double IL7Ra_IL7 = y[19];
	double gc_IL7 = y[20];
	double IL7Ra_gc_IL7 = y[21];
	double IL9R = y[22];
	double IL9R_IL9 = y[23];
	double gc_IL9 = y[24];
	double IL9R_gc_IL9 = y[25];
	
	// IL2
	dydt[0] = -r->kfbnd * IL2Ra * IL2 + r->k1rev * IL2_IL2Ra - r->kfwd * IL2Ra * IL2_gc + r->k6rev * IL2_IL2Ra_gc - r->kfwd * IL2Ra * IL2_IL2Rb_gc + r->k8rev * IL2_IL2Ra_IL2Rb_gc - r->kfwd * IL2Ra * IL2_IL2Rb + r->k12rev * IL2_IL2Ra_IL2Rb;
	dydt[1] = -r->kfbnd * IL2Rb * IL2 + r->k2rev * IL2_IL2Rb - r->kfwd * IL2Rb * IL2_gc + r->k7rev * IL2_IL2Rb_gc - r->kfwd * IL2Rb * IL2_IL2Ra_gc + r->k9rev * IL2_IL2Ra_IL2Rb_gc - r->kfwd * IL2Rb * IL2_IL2Ra + r->k11rev * IL2_IL2Ra_IL2Rb;
	dydt[2] = -r->k3fwd * IL2 * gc + r->k3rev * IL2_gc - r->kfwd * IL2_IL2Rb * gc + r->k5rev * IL2_IL2Rb_gc - r->kfwd * IL2_IL2Ra * gc + r->k4rev * IL2_IL2Ra_gc - r->kfwd * IL2_IL2Ra_IL2Rb * gc + r->k10rev * IL2_IL2Ra_IL2Rb_gc;
	dydt[3] = -r->kfwd * IL2_IL2Ra * IL2Rb + r->k11rev * IL2_IL2Ra_IL2Rb - r->kfwd * IL2_IL2Ra * gc + r->k4rev * IL2_IL2Ra_gc + r->kfbnd * IL2 * IL2Ra - r->k1rev * IL2_IL2Ra;
	dydt[4] = -r->kfwd * IL2_IL2Rb * IL2Ra + r->k12rev * IL2_IL2Ra_IL2Rb - r->kfwd * IL2_IL2Rb * gc + r->k5rev * IL2_IL2Rb_gc + r->kfbnd * IL2 * IL2Rb - r->k2rev * IL2_IL2Rb;
	dydt[5] = -r->kfwd * IL2_gc * IL2Ra + r->k6rev * IL2_IL2Ra_gc - r->kfwd * IL2_gc * IL2Rb + r->k7rev * IL2_IL2Rb_gc + r->k3fwd * IL2 * gc - r->k3rev * IL2_gc;
	dydt[6] = -r->kfwd * IL2_IL2Ra_IL2Rb * gc + r->k10rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra * IL2Rb - r->k11rev * IL2_IL2Ra_IL2Rb + r->kfwd * IL2_IL2Rb * IL2Ra - r->k12rev * IL2_IL2Ra_IL2Rb;
	dydt[7] = -r->kfwd * IL2_IL2Ra_gc * IL2Rb + r->k9rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra * gc - r->k4rev * IL2_IL2Ra_gc + r->kfwd * IL2_gc * IL2Ra - r->k6rev * IL2_IL2Ra_gc;
	dydt[8] = -r->kfwd * IL2_IL2Rb_gc * IL2Ra + r->k8rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * gc * IL2_IL2Rb - r->k5rev * IL2_IL2Rb_gc + r->kfwd * IL2_gc * IL2Rb - r->k7rev * IL2_IL2Rb_gc;
	dydt[9] = r->kfwd * IL2_IL2Rb_gc * IL2Ra - r->k8rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra_gc * IL2Rb - r->k9rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra_IL2Rb * gc - r->k10rev * IL2_IL2Ra_IL2Rb_gc;

	// IL15
	dydt[10] = -r->kfbnd * IL15Ra * IL15 + r->k13rev * IL15_IL15Ra - r->kfbnd * IL15Ra * IL15_gc + r->k18rev * IL15_IL15Ra_gc - r->kfwd * IL15Ra * IL15_IL2Rb_gc + r->k20rev * IL15_IL15Ra_IL2Rb_gc - r->kfwd * IL15Ra * IL15_IL2Rb + r->k24rev * IL15_IL15Ra_IL2Rb;
	dydt[11] = -r->kfwd * IL15_IL15Ra * IL2Rb + r->k23rev * IL15_IL15Ra_IL2Rb - r->kfwd * IL15_IL15Ra * gc + r->k16rev * IL15_IL15Ra_gc + r->kfbnd * IL15 * IL15Ra - r->k13rev * IL15_IL15Ra;
	dydt[12] = -r->kfwd * IL15_IL2Rb * IL15Ra + r->k24rev * IL15_IL15Ra_IL2Rb - r->kfbnd * IL15_IL2Rb * gc + r->k17rev * IL15_IL2Rb_gc + r->kfbnd * IL15 * IL2Rb - r->k14rev * IL15_IL2Rb;
	dydt[13] = -r->kfbnd * IL15_gc * IL15Ra + r->k18rev * IL15_IL15Ra_gc - r->kfwd * IL15_gc * IL2Rb + r->k19rev * IL15_IL2Rb_gc + r->kfbnd * IL15 * gc - r->k15rev * IL15_gc;
	dydt[14] = -r->kfwd * IL15_IL15Ra_IL2Rb * gc + r->k22rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra * IL2Rb - r->k23rev * IL15_IL15Ra_IL2Rb + r->kfwd * IL15_IL2Rb * IL15Ra - r->k24rev * IL15_IL15Ra_IL2Rb;
	dydt[15] = -r->kfwd * IL15_IL15Ra_gc * IL2Rb + r->k21rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra * gc - r->k16rev * IL15_IL15Ra_gc + r->kfbnd * IL15_gc * IL15Ra - r->k18rev * IL15_IL15Ra_gc;
	dydt[16] = -r->kfwd * IL15_IL2Rb_gc * IL15Ra + r->k20rev * IL15_IL15Ra_IL2Rb_gc + r->kfbnd * gc * IL15_IL2Rb - r->k17rev * IL15_IL2Rb_gc + r->kfwd * IL15_gc * IL2Rb - r->k19rev * IL15_IL2Rb_gc;
	dydt[17] =  r->kfwd * IL15_IL2Rb_gc * IL15Ra - r->k20rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra_gc * IL2Rb - r->k21rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra_IL2Rb * gc - r->k22rev * IL15_IL15Ra_IL2Rb_gc;
	
	dydt[1] = dydt[1] - r->kfbnd * IL2Rb * IL15 + r->k14rev * IL15_IL2Rb - r->kfwd * IL2Rb * IL15_gc + r->k19rev * IL15_IL2Rb_gc - r->kfwd * IL2Rb * IL15_IL15Ra_gc + r->k21rev * IL15_IL15Ra_IL2Rb_gc - r->kfwd * IL2Rb * IL15_IL15Ra + r->k23rev * IL15_IL15Ra_IL2Rb;
	dydt[2] = dydt[2] - r->kfbnd * IL15 * gc + r->k15rev * IL15_gc - r->kfbnd * IL15_IL2Rb * gc + r->k17rev * IL15_IL2Rb_gc - r->kfwd * IL15_IL15Ra * gc + r->k16rev * IL15_IL15Ra_gc - r->kfwd * IL15_IL15Ra_IL2Rb * gc + r->k22rev * IL15_IL15Ra_IL2Rb_gc;
	
	// IL7
	dydt[2] = dydt[2] - r->kfbnd * IL7 * gc + r->k26rev * gc_IL7 - r->kfwd * gc * IL7Ra_IL7 + r->k27rev * IL7Ra_gc_IL7;
	dydt[18] = -r->kfbnd * IL7Ra * IL7 + r->k25rev * IL7Ra_IL7 - r->kfwd * IL7Ra * gc_IL7 + r->k28rev * IL7Ra_gc_IL7;
	dydt[19] = r->kfbnd * IL7Ra * IL7 - r->k25rev * IL7Ra_IL7 - r->kfwd * gc * IL7Ra_IL7 + r->k27rev * IL7Ra_gc_IL7;
	dydt[20] = -r->kfwd * IL7Ra * gc_IL7 + r->k28rev * IL7Ra_gc_IL7 + r->kfbnd * IL7 * gc - r->k26rev * gc_IL7;
	dydt[21] = r->kfwd * IL7Ra * gc_IL7 - r->k28rev * IL7Ra_gc_IL7 + r->kfwd * gc * IL7Ra_IL7 - r->k27rev * IL7Ra_gc_IL7;

	// IL9
	dydt[2] = dydt[2] - r->kfbnd * IL9 * gc + r->k30rev * gc_IL9 - r->kfwd * gc * IL9R_IL9 + r->k31rev * IL9R_gc_IL9;
	dydt[22] = -r->kfbnd * IL9R * IL9 + r->k29rev * IL9R_IL9 - r->kfwd * IL9R * gc_IL9 + r->k32rev * IL9R_gc_IL9;
	dydt[23] = r->kfbnd * IL9R * IL9 - r->k29rev * IL9R_IL9 - r->kfwd * gc * IL9R_IL9 + r->k31rev * IL9R_gc_IL9;
	dydt[24] = -r->kfwd * IL9R * gc_IL9 + r->k32rev * IL9R_gc_IL9 + r->kfbnd * IL9 * gc - r->k30rev * gc_IL9;
	dydt[25] = r->kfwd * IL9R * gc_IL9 - r->k32rev * IL9R_gc_IL9 + r->kfwd * gc * IL9R_IL9 - r->k31rev * IL9R_gc_IL9;
}


extern "C" void dydt_C(double *y_in, double t, double *dydt_out, double *rxn_in) {
	array<double, 12> tfr;

	ratesS r = param(rxn_in, tfr.data());

	dy_dt(y_in, &r, dydt_out, r.IL2, r.IL15, r.IL7, r.IL9);
}


void findLigConsume(double *dydt) {
	// Calculate the ligand consumption.
	dydt[52] -= std::accumulate(dydt+3, dydt+10, 0) / internalV;
	dydt[53] -= std::accumulate(dydt+11, dydt+18, 0) / internalV;
	dydt[54] -= std::accumulate(dydt+19, dydt+22, 0) / internalV;
	dydt[55] -= std::accumulate(dydt+23, dydt+26, 0) / internalV;
}


void trafficking(const double * const y, const ratesS * const r, double * const dydt) {
	// Implement trafficking.
	size_t halfL = activeV.size();

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

	// Expression: IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R
	dydt[0] += r->Rexpr[0];
	dydt[1] += r->Rexpr[1];
	dydt[2] += r->Rexpr[2];
	dydt[10] += r->Rexpr[3];
	dydt[18] += r->Rexpr[4];
	dydt[22] += r->Rexpr[5];

	// Degradation does lead to some clearance of ligand in the endosome
	for (size_t ii = 0; ii < 4; ii++) {
		dydt[52 + ii] -= dydt[52 + ii] * r->kDeg;
	}
}


void fullModel(const double * const y, const ratesS * const r, double *dydt) {
	// Implement full model.
	fill(dydt, dydt + 56, 0.0);

	// Calculate cell surface and endosomal reactions
	dy_dt(y,      r, dydt,     r->IL2, r->IL15, r->IL7, r->IL9);
	dy_dt(y + 26, r, dydt + 26, y[52],   y[53],  y[54],  y[55]);

	// Handle trafficking
	trafficking(y, r, dydt);

	// Handle endosomal ligand balance.
	findLigConsume(dydt);
}


int fullModelCVode (const double, const N_Vector xx, N_Vector dxxdt, void *user_data) {
	ratesS *rIn = static_cast<ratesS *>(user_data);

	array<double, 56> xxArr;

	// Get the data in the right form
	if (NV_LENGTH_S(xx) == xxArr.size()) { // If we're using the full model
		fullModel(xxArr.data(), rIn, NV_DATA_S(dxxdt));
	} else if (NV_LENGTH_S(xx) == IL2_assoc.size()) { // If it looks like we're using the IL2 model
		fill(xxArr.begin(), xxArr.end(), 0.0);

		for (size_t ii = 0; ii < IL2_assoc.size(); ii++)
			xxArr[IL2_assoc[ii]] = NV_Ith_S(xx, ii);

		array<double, 56> dydt;

		fullModel(xxArr.data(), rIn, dydt.data());

		for (size_t ii = 0; ii < IL2_assoc.size(); ii++)
			NV_Ith_S(dxxdt, ii) = dydt[IL2_assoc[ii]];
	} else {
		throw std::runtime_error(string("Failed to find the right wrapper."));
	}

	return 0;
}


extern "C" void fullModel_C(const double * const y_in, double t, double *dydt_out, double *rxn_in, double *tfr_in) {
	// Bring back the wrapper!

	ratesS r = param(rxn_in, tfr_in);

	fullModel(y_in, &r, dydt_out);
}


array<double, 56> solveAutocrine(const ratesS * const r) {
	array<double, 56> y0;
	fill(y0.begin(), y0.end(), 0.0);

	array<size_t, 6> recIDX = {{0, 1, 2, 10, 18, 22}};

	double internalFrac = 0.5; // Same as that used in TAM model

	// Expand out trafficking terms
	double kRec = r->kRec*(1-r->sortF);
	double kDeg = r->kDeg*r->sortF;

	// Assuming no autocrine ligand, so can solve steady state
	// Add the species
	for (size_t ii = 0; ii < recIDX.size(); ii++) {
		y0[recIDX[ii] + 26] = r->Rexpr[ii] / kDeg / internalFrac;
		y0[recIDX[ii]] = (r->Rexpr[ii] + kRec*y0[recIDX[ii] + 26]*internalFrac)/r->endo;
	}

	return y0;
}


static void errorHandler(int error_code, const char *module, const char *function, char *msg, void *user_data) {
	if (error_code == CV_WARNING) return;

	std::cout << "Internal CVode error in " << function << std::endl;
	std::cout << msg << std::endl;
	std::cout << "In module: " << module << std::endl;
	std::cout << "Error code: " << error_code << std::endl;
}

void* solver_setup(N_Vector init, void * params) {
	/* Call CVodeCreate to create the solver memory and specify the
	 * Backward Differentiation Formula and the use of a Newton iteration */
	void *cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
	if (cvode_mem == NULL) {
		CVodeFree(&cvode_mem);
		throw std::runtime_error(string("Error calling CVodeCreate in solver_setup."));
	}
	
	CVodeSetErrHandlerFn(cvode_mem, &errorHandler, params);

	/* Call CVodeInit to initialize the integrator memory and specify the
	 * user's right hand side function in y'=f(t,y), the inital time T0, and
	 * the initial dependent variable vector y. */
	if (CVodeInit(cvode_mem, fullModelCVode, 0.0, init) < 0) {
		CVodeFree(&cvode_mem);
		throw std::runtime_error(string("Error calling CVodeInit in solver_setup."));
	}
	
	N_Vector abbstol = N_VNew_Serial(NV_LENGTH_S(init));
	N_VConst(abstolIn, abbstol);
	
	/* Call CVodeSVtolerances to specify the scalar relative tolerance
	 * and vector absolute tolerances */
	if (CVodeSVtolerances(cvode_mem, reltolIn, abbstol) < 0) {
		N_VDestroy_Serial(abbstol);
		CVodeFree(&cvode_mem);
		throw std::runtime_error(string("Error calling CVodeSVtolerances in solver_setup."));
	}
	N_VDestroy_Serial(abbstol);

	SUNMatrix A = SUNDenseMatrix((int) NV_LENGTH_S(init), (int) NV_LENGTH_S(init));

	SUNLinearSolver LS = SUNDenseLinearSolver(init, A);
	
	// Call CVDense to specify the CVDENSE dense linear solver
	if (CVDlsSetLinearSolver(cvode_mem, LS, A) < 0) {
		CVodeFree(&cvode_mem);
		throw std::runtime_error(string("Error calling CVDense in solver_setup."));
	}
	
	// Pass along the parameter structure to the differential equations
	if (CVodeSetUserData(cvode_mem, params) < 0) {
		CVodeFree(&cvode_mem);
		throw std::runtime_error(string("Error calling CVodeSetUserData in solver_setup."));
	}

	CVodeSetMaxNumSteps(cvode_mem, 2000000);
	
	return cvode_mem;
}


extern "C" int runCkine (double *tps, size_t ntps, double *out, double *rxnRatesIn, double *trafRatesIn) {
	ratesS rattes = param(rxnRatesIn, trafRatesIn);

	array<double, 56> y0 = solveAutocrine(&rattes);
	N_Vector state;

	// Fill output values with 0's
	fill(out, out + ntps*y0.size(), 0.0);

	// Can we use the reduced IL2 only model
	if (rattes.Rexpr[3] + rattes.Rexpr[4] + rattes.Rexpr[5] == 0.0) {
		state = N_VNew_Serial((long) IL2_assoc.size());

		for (size_t ii = 0; ii < IL2_assoc.size(); ii++)
			NV_Ith_S(state, ii) = y0[IL2_assoc[ii]];
	} else { // Just the full model
		state = N_VMake_Serial((long) y0.size(), y0.data());
	}

	void *cvode_mem = solver_setup(state, (void *) &rattes);

	double tret = 0;

	for (size_t itps = 0; itps < ntps; itps++) {
		if (tps[itps] < tret) {
			std::cout << "Can't go backwards." << std::endl;
			N_VDestroy_Serial(state);
			CVodeFree(&cvode_mem);
			return -1;
		}

		int returnVal = CVode(cvode_mem, tps[itps], state, &tret, CV_NORMAL);
		
		if (returnVal < 0) {
			std::cout << "CVode error in CVode. Code: " << returnVal << std::endl;
			N_VDestroy_Serial(state);
			CVodeFree(&cvode_mem);
			return returnVal;
		}

		// Copy out result
		if (NV_LENGTH_S(state) == y0.size()) {
			copy_n(NV_DATA_S(state), y0.size(), out + y0.size()*itps);
		} else { // If we're dealing with a reduced model
			for (size_t ii = 0; ii < IL2_assoc.size(); ii++) {
				out[y0.size()*itps + IL2_assoc[ii]] = NV_Ith_S(state, ii);
			}
		}
	}

	N_VDestroy_Serial(state);
	CVodeFree(&cvode_mem);
	return 0;
}
