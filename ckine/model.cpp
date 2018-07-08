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
#include <cvodes/cvodes.h>             /* prototypes for CVODE fcts., consts.  */
#include <cvode/cvode_direct.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "model.hpp"

using std::array;
using std::copy;
using std::vector;
using std::fill;
using std::string;

typedef Eigen::Matrix<double, Nspecies, Nspecies, Eigen::RowMajor> JacMat;

int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data, N_Vector, N_Vector, N_Vector);

const array<size_t, 6> recIDX = {{0, 1, 2, 9, 16, 19}};

std::array<bool, halfL> __active_species_IDX() {
	std::array<bool, halfL> __active_species_IDX;
	std::fill(__active_species_IDX.begin(), __active_species_IDX.end(), false);

	__active_species_IDX[7] = true;
	__active_species_IDX[8] = true;
	__active_species_IDX[14] = true;
	__active_species_IDX[15] = true;
	__active_species_IDX[18] = true;
	__active_species_IDX[21] = true;

	return __active_species_IDX;
}

const std::array<bool, halfL> activeV = __active_species_IDX();

ratesS param(const double * const rxntfR) {
	ratesS r;

	r.IL2 = rxntfR[0];
	r.IL15 = rxntfR[1];
	r.IL7 = rxntfR[2];
	r.IL9 = rxntfR[3];
	r.kfwd = rxntfR[4];
	r.k4rev = rxntfR[5];
	r.k5rev = rxntfR[6];
	r.k16rev = rxntfR[7];
	r.k17rev = rxntfR[8];
	r.k22rev = rxntfR[9];
	r.k23rev = rxntfR[10];
	r.k27rev = rxntfR[11];
	r.k31rev = rxntfR[12];

	// These are probably measured in the literature
	r.k10rev = 12.0 * r.k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
	r.k11rev = 63.0 * r.k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
	// To satisfy detailed balance these relationships should hold
	// Based on initial assembly steps
	r.k12rev = k1rev * r.k11rev / k2rev; // loop for IL2_IL2Ra_IL2Rb
	// Based on formation of full complex (IL2_IL2Ra_IL2Rb_gc)
	r.k9rev = r.k10rev * r.k11rev / r.k4rev;
	r.k8rev = r.k10rev * r.k12rev / r.k5rev;

	// IL15
	// To satisfy detailed balance these relationships should hold
	// _Based on initial assembly steps
	r.k24rev = k13rev * r.k23rev / k14rev; // loop for IL15_IL15Ra_IL2Rb still holds

	// _Based on formation of full complex
	r.k21rev = r.k22rev * r.k23rev / r.k16rev;
	r.k20rev = r.k22rev * r.k24rev / r.k17rev;

	// Set the rates
	r.endo = rxntfR[13];
	r.activeEndo = rxntfR[14];
	r.sortF = rxntfR[15];
	r.kRec = rxntfR[16];
	r.kDeg = rxntfR[17];

	if (r.sortF > 1.0) {
		throw std::runtime_error(string("sortF is a fraction and cannot be greater than 1.0."));
	}

	std::copy_n(rxntfR + 18, 6, r.Rexpr.begin());

	return r;
}

void dy_dt(const double * const y, const ratesS * const r, double * const dydt, double IL2, double IL15, double IL7, double IL9) {
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
	
	// IL7, IL9 in nM
	const double IL7Ra = y[16];
	const double IL7Ra_IL7 = y[17];
	const double IL7Ra_gc_IL7 = y[18];
	const double IL9R = y[19];
	const double IL9R_IL9 = y[20];
	const double IL9R_gc_IL9 = y[21];
	
	// IL2
	dydt[0] = -kfbnd * IL2Ra * IL2 + k1rev * IL2_IL2Ra - r->kfwd * IL2Ra * IL2_IL2Rb_gc + r->k8rev * IL2_IL2Ra_IL2Rb_gc - r->kfwd * IL2Ra * IL2_IL2Rb + r->k12rev * IL2_IL2Ra_IL2Rb;
	dydt[1] = -kfbnd * IL2Rb * IL2 + k2rev * IL2_IL2Rb - r->kfwd * IL2Rb * IL2_IL2Ra_gc + r->k9rev * IL2_IL2Ra_IL2Rb_gc - r->kfwd * IL2Rb * IL2_IL2Ra + r->k11rev * IL2_IL2Ra_IL2Rb;
	dydt[2] = -r->kfwd * IL2_IL2Rb * gc + r->k5rev * IL2_IL2Rb_gc - r->kfwd * IL2_IL2Ra * gc + r->k4rev * IL2_IL2Ra_gc - r->kfwd * IL2_IL2Ra_IL2Rb * gc + r->k10rev * IL2_IL2Ra_IL2Rb_gc;
	dydt[3] = -r->kfwd * IL2_IL2Ra * IL2Rb + r->k11rev * IL2_IL2Ra_IL2Rb - r->kfwd * IL2_IL2Ra * gc + r->k4rev * IL2_IL2Ra_gc + kfbnd * IL2 * IL2Ra - k1rev * IL2_IL2Ra;
	dydt[4] = -r->kfwd * IL2_IL2Rb * IL2Ra + r->k12rev * IL2_IL2Ra_IL2Rb - r->kfwd * IL2_IL2Rb * gc + r->k5rev * IL2_IL2Rb_gc + kfbnd * IL2 * IL2Rb - k2rev * IL2_IL2Rb;
	dydt[5] = -r->kfwd * IL2_IL2Ra_IL2Rb * gc + r->k10rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra * IL2Rb - r->k11rev * IL2_IL2Ra_IL2Rb + r->kfwd * IL2_IL2Rb * IL2Ra - r->k12rev * IL2_IL2Ra_IL2Rb;
	dydt[6] = -r->kfwd * IL2_IL2Ra_gc * IL2Rb + r->k9rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra * gc - r->k4rev * IL2_IL2Ra_gc;
	dydt[7] = -r->kfwd * IL2_IL2Rb_gc * IL2Ra + r->k8rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * gc * IL2_IL2Rb - r->k5rev * IL2_IL2Rb_gc;
	dydt[8] = r->kfwd * IL2_IL2Rb_gc * IL2Ra - r->k8rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra_gc * IL2Rb - r->k9rev * IL2_IL2Ra_IL2Rb_gc + r->kfwd * IL2_IL2Ra_IL2Rb * gc - r->k10rev * IL2_IL2Ra_IL2Rb_gc;

	// IL15
	dydt[9] = -kfbnd * IL15Ra * IL15 + k13rev * IL15_IL15Ra - r->kfwd * IL15Ra * IL15_IL2Rb_gc + r->k20rev * IL15_IL15Ra_IL2Rb_gc - r->kfwd * IL15Ra * IL15_IL2Rb + r->k24rev * IL15_IL15Ra_IL2Rb;
	dydt[10] = -r->kfwd * IL15_IL15Ra * IL2Rb + r->k23rev * IL15_IL15Ra_IL2Rb - r->kfwd * IL15_IL15Ra * gc + r->k16rev * IL15_IL15Ra_gc + kfbnd * IL15 * IL15Ra - k13rev * IL15_IL15Ra;
	dydt[11] = -r->kfwd * IL15_IL2Rb * IL15Ra + r->k24rev * IL15_IL15Ra_IL2Rb - r->kfwd * IL15_IL2Rb * gc + r->k17rev * IL15_IL2Rb_gc + kfbnd * IL15 * IL2Rb - k14rev * IL15_IL2Rb;
	dydt[12] = -r->kfwd * IL15_IL15Ra_IL2Rb * gc + r->k22rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra * IL2Rb - r->k23rev * IL15_IL15Ra_IL2Rb + r->kfwd * IL15_IL2Rb * IL15Ra - r->k24rev * IL15_IL15Ra_IL2Rb;
	dydt[13] = -r->kfwd * IL15_IL15Ra_gc * IL2Rb + r->k21rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra * gc - r->k16rev * IL15_IL15Ra_gc;
	dydt[14] = -r->kfwd * IL15_IL2Rb_gc * IL15Ra + r->k20rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * gc * IL15_IL2Rb - r->k17rev * IL15_IL2Rb_gc;
	dydt[15] =  r->kfwd * IL15_IL2Rb_gc * IL15Ra - r->k20rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra_gc * IL2Rb - r->k21rev * IL15_IL15Ra_IL2Rb_gc + r->kfwd * IL15_IL15Ra_IL2Rb * gc - r->k22rev * IL15_IL15Ra_IL2Rb_gc;
	
	dydt[1] = dydt[1] - kfbnd * IL2Rb * IL15 + k14rev * IL15_IL2Rb - r->kfwd * IL2Rb * IL15_IL15Ra_gc + r->k21rev * IL15_IL15Ra_IL2Rb_gc - r->kfwd * IL2Rb * IL15_IL15Ra + r->k23rev * IL15_IL15Ra_IL2Rb;
	dydt[2] = dydt[2] - r->kfwd * IL15_IL2Rb * gc + r->k17rev * IL15_IL2Rb_gc - r->kfwd * IL15_IL15Ra * gc + r->k16rev * IL15_IL15Ra_gc - r->kfwd * IL15_IL15Ra_IL2Rb * gc + r->k22rev * IL15_IL15Ra_IL2Rb_gc; 
	
	// IL7
	dydt[2] = dydt[2] - r->kfwd * gc * IL7Ra_IL7 + r->k27rev * IL7Ra_gc_IL7;
	dydt[16] = -kfbnd * IL7Ra * IL7 + k25rev * IL7Ra_IL7;
	dydt[17] = kfbnd * IL7Ra * IL7 - k25rev * IL7Ra_IL7 - r->kfwd * gc * IL7Ra_IL7 + r->k27rev * IL7Ra_gc_IL7;
	dydt[18] = r->kfwd * gc * IL7Ra_IL7 - r->k27rev * IL7Ra_gc_IL7;

	// IL9
	dydt[2] = dydt[2] - r->kfwd * gc * IL9R_IL9 + r->k31rev * IL9R_gc_IL9;
	dydt[19] = -kfbnd * IL9R * IL9 + k29rev * IL9R_IL9;
	dydt[20] = kfbnd * IL9R * IL9 - k29rev * IL9R_IL9 - r->kfwd * gc * IL9R_IL9 + r->k31rev * IL9R_gc_IL9;
	dydt[21] = r->kfwd * gc * IL9R_IL9 - r->k31rev * IL9R_gc_IL9;
}


extern "C" void dydt_C(double *y_in, double, double *dydt_out, double *rxn_in) {
	ratesS r = param(rxn_in);

	dy_dt(y_in, &r, dydt_out, r.IL2, r.IL15, r.IL7, r.IL9);
}


/**
 * @brief      Solve for the ligand consumption rate in the endosome.
 *
 * @param      dydt  The rate of change vector solved for the receptor species.
 */
void findLigConsume(double *dydt) {
	double const * const dydti = dydt + halfL;

	// Calculate the ligand consumption.
	dydt[44] -= std::accumulate(dydti+3,  dydti+9, (double) 0.0) / internalV;
	dydt[45] -= std::accumulate(dydti+10, dydti+16, (double) 0.0) / internalV;
	dydt[46] -= std::accumulate(dydti+17, dydti+19, (double) 0.0) / internalV;
	dydt[47] -= std::accumulate(dydti+20, dydti+22, (double) 0.0) / internalV;
}


void trafficking(const double * const y, const ratesS * const r, double * const dydt) {
	// Implement trafficking.

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
	dydt[9] += r->Rexpr[3];
	dydt[16] += r->Rexpr[4];
	dydt[19] += r->Rexpr[5];

	// Degradation does lead to some clearance of ligand in the endosome
	for (size_t ii = 0; ii < 4; ii++) {
		dydt[44 + ii] -= y[44 + ii] * r->kDeg;
	}
}


void fullModel(const double * const y, const ratesS * const r, double *dydt) {
	// Implement full model.
	fill(dydt, dydt + Nspecies, 0.0);

	// Calculate cell surface and endosomal reactions
	dy_dt(y,      r, dydt,     r->IL2, r->IL15, r->IL7, r->IL9);
	dy_dt(y + halfL, r, dydt + halfL, y[44],   y[45],  y[46],  y[47]);

	// Handle endosomal ligand balance.
	// Must come before trafficking as we only calculate this based on reactions balance
	findLigConsume(dydt);

	// Handle trafficking
	trafficking(y, r, dydt);
}


int fullModelCVode (const double, const N_Vector xx, N_Vector dxxdt, void *user_data) {
	ratesS rattes = param(static_cast<double *>(user_data));

	// Get the data in the right form
	fullModel(NV_DATA_S(xx), &rattes, NV_DATA_S(dxxdt));

	return 0;
}


extern "C" void fullModel_C(const double * const y_in, double, double *dydt_out, double *rxn_in) {
	ratesS r = param(rxn_in);

	fullModel(y_in, &r, dydt_out);
}


array<double, Nspecies> solveAutocrine(const ratesS * const r) {
	array<double, Nspecies> y0;
	fill(y0.begin(), y0.end(), 0.0);

	// Expand out trafficking terms
	double kRec = r->kRec*(1-r->sortF);
	double kDeg = r->kDeg*r->sortF;

	// Assuming no autocrine ligand, so can solve steady state
	// Add the species
	for (size_t ii = 0; ii < recIDX.size(); ii++) {
		y0[recIDX[ii] + 22] = r->Rexpr[ii] / kDeg / internalFrac;
		y0[recIDX[ii]] = (r->Rexpr[ii] + kRec*y0[recIDX[ii] + 22]*internalFrac)/r->endo;
	}

	return y0;
}


/**
 * @brief      Setup the autocrine state sensitivities.
 *
 * @param[in]  r     Rate parameters.
 * @param      y0s   The autocrine state sensitivities.
 */
void solveAutocrineS (const ratesS * const r, N_Vector *y0s, array<double, 48> &y0) {
	for (size_t is = 0; is < Nparams; is++)
		N_VConst(0.0, y0s[is]);

	for (size_t is : recIDX) {
		// Endosomal amount doesn't depend on endo
		NV_Ith_S(y0s[13], is) = -y0[is]/r->endo; // Endo (13)

		// sortF (15)
		NV_Ith_S(y0s[15], is + 22) = -y0[is + 22]/r->sortF;
		NV_Ith_S(y0s[15], is) = r->kRec*internalFrac/r->endo*((1 - r->sortF)*NV_Ith_S(y0s[15], is + 22) - y0[is + 22]);

		// Endosomal amount doesn't depend on kRec
		NV_Ith_S(y0s[16], is) = (1-r->sortF)*y0[is + 22]*internalFrac/r->endo; // kRec (16)

		// kDeg (17)
		NV_Ith_S(y0s[17], is + 22) = -y0[is + 22]/r->kDeg;
		NV_Ith_S(y0s[17], is) = r->kRec*(1-r->sortF)*NV_Ith_S(y0s[17], is + 22)*internalFrac/r->endo;
	}

	// Rexpr (18-23)
	for (size_t ii = 0; ii < recIDX.size(); ii++) {
		NV_Ith_S(y0s[18 + ii], recIDX[ii] + 22) = y0[recIDX[ii] + 22]/r->Rexpr[ii];
		NV_Ith_S(y0s[18 + ii], recIDX[ii]) = 1/r->endo + NV_Ith_S(y0s[18 + ii], recIDX[ii] + 22)*r->kRec*(1-r->sortF)*internalFrac/r->endo;
	}
}


struct solver {
	void *cvode_mem;
	SUNLinearSolver LS;
	N_Vector state;
	N_Vector *yS;
	SUNMatrix A;
	bool sensi;
	double *params;
};


static void errorHandler(int error_code, const char *module, const char *function, char *msg, void *ehdata) {
	if (error_code == CV_WARNING) return;
	solver *sMem = static_cast<solver *>(ehdata);

	std::cout << "Internal CVode error in " << function << ", module: " << module << ", error code: " << error_code << std::endl;
	std::cout << msg << std::endl;
	std::cout << "Parameters: ";

	for (size_t ii = 0; ii < Nparams; ii++) {
		std::cout << sMem->params[ii] << "\t";
	}

	ratesS ratt = param(sMem->params);

	std::cout << "IL2: " << ratt.IL2 << std::endl;
	std::cout << "IL15: " << ratt.IL15 << std::endl;
	std::cout << "IL7: " << ratt.IL7 << std::endl;
	std::cout << "IL9: " << ratt.IL9 << std::endl;
	std::cout << "kfwd: " << ratt.kfwd << std::endl;
	std::cout << "k4rev: " << ratt.k4rev << std::endl;
	std::cout << "k5rev: " << ratt.k5rev << std::endl;
	std::cout << "k8rev: " << ratt.k8rev << std::endl;
	std::cout << "k9rev: " << ratt.k9rev << std::endl;
	std::cout << "k10rev: " << ratt.k10rev << std::endl;
	std::cout << "k11rev: " << ratt.k11rev << std::endl;
	std::cout << "k12rev: " << ratt.k12rev << std::endl;
	std::cout << "k16rev: " << ratt.k16rev << std::endl;
	std::cout << "k17rev: " << ratt.k17rev << std::endl;
	std::cout << "k20rev: " << ratt.k20rev << std::endl;
	std::cout << "k21rev: " << ratt.k21rev << std::endl;
	std::cout << "k22rev: " << ratt.k22rev << std::endl;
	std::cout << "k23rev: " << ratt.k23rev << std::endl;
	std::cout << "k24rev: " << ratt.k24rev << std::endl;
	std::cout << "k27rev: " << ratt.k27rev << std::endl;
	std::cout << "k31rev: " << ratt.k31rev << std::endl;
	std::cout << "endo: " << ratt.endo << std::endl;
	std::cout << "activeEndo: " << ratt.activeEndo << std::endl;
	std::cout << "sortF: " << ratt.sortF << std::endl;
	std::cout << "kRec: " << ratt.kRec << std::endl;
	std::cout << "kDeg: " << ratt.kDeg << std::endl;

	std::cout << "Rexpr 1: " << ratt.Rexpr[0] << std::endl;
	std::cout << "Rexpr 2: " << ratt.Rexpr[1] << std::endl;
	std::cout << "Rexpr 3: " << ratt.Rexpr[2] << std::endl;
	std::cout << "Rexpr 4: " << ratt.Rexpr[3] << std::endl;
	std::cout << "Rexpr 5: " << ratt.Rexpr[4] << std::endl;
	std::cout << "Rexpr 6: " << ratt.Rexpr[5] << std::endl;

	std::cout << std::endl;

	if (sMem->sensi)
		std::cout << "Sensitivity enabled." << std::endl;

	std::cout << std::endl << std::endl;
}


void solverFree(solver *sMem) {
	if (sMem->sensi) {
		CVodeSensFree(sMem->cvode_mem);
		N_VDestroyVectorArray(sMem->yS, Nparams);
	}

	N_VDestroy_Serial(sMem->state);
	CVodeFree(&sMem->cvode_mem);
	SUNLinSolFree(sMem->LS);
	SUNMatDestroy(sMem->A);
}


void solver_setup(solver *sMem, double *params) {
	// So far we're not doing a sensitivity analysis
	sMem->sensi = false;
	sMem->params = params;

	/* Call CVodeCreate to create the solver memory and specify the
	 * Backward Differentiation Formula and the use of a Newton iteration */
	sMem->cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
	if (sMem->cvode_mem == nullptr) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeCreate in solver_setup."));
	}
	
	CVodeSetErrHandlerFn(sMem->cvode_mem, &errorHandler, static_cast<void *>(sMem));

	/* Call CVodeInit to initialize the integrator memory and specify the
	 * user's right hand side function in y'=f(t,y), the inital time T0, and
	 * the initial dependent variable vector y. */
	if (CVodeInit(sMem->cvode_mem, fullModelCVode, 0.0, sMem->state) < 0) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeInit in solver_setup."));
	}
	
	// Call CVodeSVtolerances to specify the scalar relative and absolute tolerances
	if (CVodeSStolerances(sMem->cvode_mem, tolIn, tolIn) < 0) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeSStolerances in solver_setup."));
	}

	sMem->A = SUNDenseMatrix(NV_LENGTH_S(sMem->state), NV_LENGTH_S(sMem->state));
	sMem->LS = SUNDenseLinearSolver(sMem->state, sMem->A);
	
	// Call CVDense to specify the CVDENSE dense linear solver
	if (CVDlsSetLinearSolver(sMem->cvode_mem, sMem->LS, sMem->A) < 0) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVDlsSetLinearSolver in solver_setup."));
	}

	CVDlsSetJacFn(sMem->cvode_mem, Jac);
	
	// Pass along the parameter structure to the differential equations
	if (CVodeSetUserData(sMem->cvode_mem, static_cast<void *>(params)) < 0) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeSetUserData in solver_setup."));
	}

	CVodeSetMaxNumSteps(sMem->cvode_mem, 800000);
}


void solver_setup_sensi(solver *sMem, const ratesS * const rr, double *params, array<double, 48> &y0) {
	// Now we are doing a sensitivity analysis
	sMem->sensi = true;

	// Set sensitivity initial conditions
	sMem->yS = N_VCloneVectorArray(Nparams, sMem->state);
	solveAutocrineS(rr, sMem->yS, y0);

	// Call CVodeSensInit1 to activate forward sensitivity computations
	// and allocate internal memory for CVODES related to sensitivity
	// calculations. Computes the right-hand sides of the sensitivity
	// ODE, one at a time
	if (CVodeSensInit(sMem->cvode_mem, Nparams, CV_SIMULTANEOUS, nullptr, sMem->yS) < 0) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeSensInit in solver_setup."));
	}

	array<double, Nparams> abs;
	fill(abs.begin(), abs.end(), tolIn);

	// Call CVodeSensSStolerances to estimate tolerances for sensitivity 
	// variables based on the rolerances supplied for states variables and 
	// the scaling factor pbar
	if (CVodeSensSStolerances(sMem->cvode_mem, tolIn, abs.data()) < 0) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeSensSStolerances in solver_setup."));
	}

	array<double, Nparams> paramArr;
	array<int, Nparams> paramList;
	std::copy_n(params, Nparams, paramArr.begin());
	for(size_t is = 0; is < Nparams; is++) {
		paramList[is] = static_cast<int>(is);

		if (paramArr[is] < std::numeric_limits<double>::epsilon())
			paramArr[is] = 0.1;
	}

	// Specify problem parameter information for sensitivity calculations
	if (CVodeSetSensParams(sMem->cvode_mem, params, paramArr.data(), paramList.data()) < 0) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeSetSensParams in solver_setup."));
	}
}


void copyOutSensi(double *out, solver *sMem) {
	for (size_t ii = 0; ii < Nparams; ii++) {
		std::copy_n(NV_DATA_S(sMem->yS[ii]), Nspecies, out + ii*Nspecies);
	}
}


extern "C" int runCkine (double *tps, size_t ntps, double *out, double *rxnRatesIn, bool sensi, double *sensiOut) {
	ratesS rattes = param(rxnRatesIn);
	size_t itps = 0;

	array<double, Nspecies> y0 = solveAutocrine(&rattes);

	solver sMem;

	// Just the full model
	sMem.state = N_VMake_Serial(static_cast<long>(Nspecies), y0.data());

	solver_setup(&sMem, rxnRatesIn);

	if (sensi) solver_setup_sensi(&sMem, &rattes, rxnRatesIn, y0);

	double tret = 0;

	if (tps[0] < std::numeric_limits<double>::epsilon()) {
		std::copy_n(y0.begin(), y0.size(), out);

		if (sensi) copyOutSensi(sensiOut, &sMem);

		itps = 1;
	}

	for (; itps < ntps; itps++) {
		if (tps[itps] < tret) {
			std::cout << "Can't go backwards." << std::endl;
			solverFree(&sMem);
			return -1;
		}

		int returnVal = CVode(sMem.cvode_mem, tps[itps], sMem.state, &tret, CV_NORMAL);
		
		if (returnVal < 0) {
			std::cout << "CVode error in CVode. Code: " << returnVal << std::endl;
			solverFree(&sMem);
			return returnVal;
		}

		// Copy out result
		std::copy_n(NV_DATA_S(sMem.state), y0.size(), out + y0.size()*itps);

		if (sensi) {
			CVodeGetSens(sMem.cvode_mem, &tps[itps], sMem.yS);
			copyOutSensi(sensiOut + Nspecies*Nparams*itps, &sMem);
		}
	}

	solverFree(&sMem);
	return 0;
}


void jacobian(const double * const y, const ratesS * const r, double * const dydt, double IL2, double IL15, double IL7, double IL9) {
	// IL2 in nM
	const double IL2Ra = y[0];
	const double IL2Rb = y[1];
	const double gc = y[2];
	const double IL2_IL2Ra = y[3];
	const double IL2_IL2Rb = y[4];
	const double IL2_IL2Ra_IL2Rb = y[5];
	const double IL2_IL2Ra_gc = y[6];
	const double IL2_IL2Rb_gc = y[7];
	
	// IL15 in nM
	const double IL15Ra = y[9];
	const double IL15_IL15Ra = y[10];
	const double IL15_IL2Rb = y[11];
	const double IL15_IL15Ra_IL2Rb = y[12];
	const double IL15_IL15Ra_gc = y[13];
	const double IL15_IL2Rb_gc = y[14];
	
	// IL7, IL9 in nM
	const double IL7Ra_IL7 = y[17];
	const double IL9R_IL9 = y[20];

	Eigen::Map<Eigen::Matrix<double, halfL, halfL, Eigen::RowMajor>> out(dydt);
	
	// unless otherwise specified, assume all partial derivatives are 0
	out.setConstant(0.0);
		
	// IL2Ra
	out(0, 0) = -kfbnd * IL2 - r->kfwd * IL2_IL2Rb_gc - r->kfwd * IL2_IL2Rb; // IL2Ra with respect to IL2Ra
	out(0, 3) = k1rev; // IL2Ra with respect to IL2_IL2Ra
	out(0, 4) = -r->kfwd * IL2Ra; // IL2Ra with respect to IL2_IL2Rb
	out(0, 5) = r->k12rev; // IL2Ra with respect to IL2_IL2Ra_IL2Rb
	out(0, 7) = -r->kfwd * IL2Ra; // IL2Ra with respect to IL2_IL2Rb_gc
	out(0, 8) = r->k8rev; // IL2Ra with respect to IL2_IL2Ra_IL2Rb_gc
	
	// IL2Rb
	out(1, 1) = -kfbnd * IL2 - r->kfwd * IL2_IL2Ra_gc - r->kfwd * IL2_IL2Ra - kfbnd * IL15 - r->kfwd * IL15_IL15Ra_gc - r->kfwd * IL15_IL15Ra; // partial derivative of IL2Rb with respect to IL2Rb
	out(1, 3) = - r->kfwd * IL2Rb; // IL2Rb with respect to IL2_IL2Ra
	out(1, 4) = k2rev; // IL2Rb with respect to IL2_IL2Rb
	out(1, 5) = r->k11rev; // IL2Rb with respect to IL2_IL2Ra_IL2Rb
	out(1, 6) = - r->kfwd * IL2Rb; // IL2Rb with respect to IL2_IL2Ra_gc
	out(1, 8) = r->k9rev; // IL2Rb with respect to IL2_IL2Ra_IL2Rb_gc
	out(1, 10) = - r->kfwd * IL2Rb; // IL2Rb with respect to IL15_IL15Ra
	out(1, 11) = k14rev; // IL2Rb with respect to IL15_IL2Rb
	out(1, 12) = r->k23rev; // IL2Rb with respect to IL15_IL15Ra_IL2Rb
	out(1, 13) = - r->kfwd * IL2Rb; // IL2Rb with respect to IL15_IL15Ra_gc
	out(1, 15) = r->k21rev; // IL2Rb with respect to IL15_IL15Ra_IL2Rb_gc
	
	// gc    
	out(2, 2) = - r->kfwd * IL2_IL2Rb - r->kfwd * IL2_IL2Ra - r->kfwd * IL2_IL2Ra_IL2Rb - r->kfwd * IL15_IL2Rb - r->kfwd * IL15_IL15Ra - r->kfwd * IL15_IL15Ra_IL2Rb - r->kfwd * IL7Ra_IL7 - r->kfwd * IL9R_IL9; // gc with respect to gc
	out(2, 3) = - r->kfwd * gc; // gc with respect to IL2_IL2Ra
	out(2, 4) = - r->kfwd * gc; // gc with respect to IL2_IL2Rb
	out(2, 5) = - r->kfwd * gc; // gc with respect to IL2_IL2Ra_IL2Rb
	out(2, 6) = r->k4rev; // gc with respect to IL2_IL2Ra_gc
	out(2, 7) = r->k5rev; // gc with respect to IL2_IL2Rb_gc
	out(2, 8) = r->k10rev; // gc with respect to IL2_IL2Ra_IL2Rb_gc
	out(2, 10) = - r->kfwd * gc; // gc with respect to IL15_IL15Ra
	out(2, 11) = - r->kfwd * gc; // gc with respect to IL15_IL2Rb
	out(2, 12) = - r->kfwd * gc; // gc with respect to IL15_IL15Ra_IL2Rb
	out(2, 13) = r->k16rev; // gc with respect to IL15_IL15Ra_gc
	out(2, 14) = r->k17rev; // gc with respect to IL15_IL2Rb_gc
	out(2, 15) = r->k22rev; // gc with respect to IL15_IL15Ra_IL2Rb_gc
	out(2, 17) = - r->kfwd * gc; // gc with respect to IL7Ra_IL7
	out(2, 18) = r->k27rev; // gc with respect to IL7Ra_gc_IL7
	out(2, 20) = - r->kfwd * gc; // gc with respect to IL9R_IL9
	out(2, 21) = r->k31rev; // gc with respect to IL9R_gc_IL9
	
	// IL2_IL2Ra
	out(3, 0) = kfbnd * IL2; // IL2_IL2Ra with respect to IL2Ra
	out(3, 1) = -r->kfwd * IL2_IL2Ra; // IL2_IL2Ra with respect to IL2Rb
	out(3, 2) = - r->kfwd * IL2_IL2Ra; // IL2_IL2Ra with respect to gc
	out(3, 3) = -r->kfwd * IL2Rb - r->kfwd * gc - k1rev; // IL2_IL2Ra with respect to IL2_IL2Ra
	out(3, 5) = r->k11rev; // IL2_IL2Ra with respect to IL2_IL2Ra_IL2Rb
	out(3, 6) = r->k4rev; // IL2_IL2Ra with respect to IL2_IL2Ra_gc
	
	// IL2_IL2Rb
	out(4, 0) = -r->kfwd * IL2_IL2Rb; // IL2_IL2Rb with respect to IL2Ra
	out(4, 1) = kfbnd * IL2; // IL2_IL2Rb with respect to IL2Rb
	out(4, 2) = - r->kfwd * IL2_IL2Rb; // IL2_IL2Rb with respect to gc
	out(4, 4) = -r->kfwd * IL2Ra - r->kfwd * gc - k2rev; // IL2_IL2Rb with respect to IL2_IL2Rb
	out(4, 5) = r->k12rev; // IL2_IL2Rb with respect to IL2_IL2Ra_IL2Rb
	out(4, 7) = r->k5rev; // IL2_IL2Rb with respect to IL2_IL2Rb_gc
	
	// IL2_IL2Ra_IL2Rb
	out(5, 0) = r->kfwd * IL2_IL2Rb; // IL2_IL2Ra_IL2Rb with respect to IL2Ra
	out(5, 1) = r->kfwd * IL2_IL2Ra; // IL2_IL2Ra_IL2Rb with respect to IL2Rb
	out(5, 2) = -r->kfwd * IL2_IL2Ra_IL2Rb; // IL2_IL2Ra_IL2Rb with respect to gc
	out(5, 3) = r->kfwd * IL2Rb; // IL2_IL2Ra_IL2Rb with respect to IL2_IL2Ra
	out(5, 4) = r->kfwd * IL2Ra; // IL2_IL2Ra_IL2Rb with respect to IL2_IL2Rb
	out(5, 5) = -r->kfwd * gc - r->k11rev - r->k12rev; // IL2_IL2Ra_IL2Rb with respect to IL2_IL2Ra_IL2Rb
	out(5, 8) = r->k10rev; // IL2_IL2Ra_IL2Rb with respect to IL2_IL2Ra_IL2Rb_gc
	
	// IL2_IL2Ra_gc
	out(6, 1) = -r->kfwd * IL2_IL2Ra_gc; // IL2_IL2Ra_gc with respect to IL2Rb
	out(6, 2) = r->kfwd * IL2_IL2Ra; // IL2_IL2Ra_gc with respect to gc
	out(6, 3) = r->kfwd * gc; // IL2_IL2Ra_gc with respect to IL2_IL2Ra
	out(6, 6) = -r->kfwd * IL2Rb - r->k4rev; // IL2_IL2Ra_gc with respect to IL2_IL2Ra_gc
	out(6, 8) = r->k9rev; // IL2_IL2Ra_gc with respect to IL2_IL2Ra_IL2Rb_gc
	
	// IL2_IL2Rb_gc
	out(7, 0) = -r->kfwd * IL2_IL2Rb_gc; // IL2_IL2Rb_gc with respect to IL2Ra
	out(7, 2) = r->kfwd * IL2_IL2Rb; // IL2_IL2Rb_gc with respect to gc
	out(7, 4) = r->kfwd * gc; // IL2_IL2Rb_gc with respect to IL2_IL2Rb
	out(7, 7) = -r->kfwd * IL2Ra - r->k5rev; // IL2_IL2Rb_gc with respect to IL2_IL2Rb_gc
	out(7, 8) = r->k8rev; // IL2_IL2Rb_gc with respect to IL2_IL2Ra_IL2Rb_gc
	
	// IL2_IL2Ra_IL2Rb_gc
	out(8, 0) = r->kfwd * IL2_IL2Rb_gc; // IL2_IL2Ra_IL2Rb_gc with respect to IL2Ra
	out(8, 1) = r->kfwd * IL2_IL2Ra_gc; // IL2_IL2Ra_IL2Rb_gc with respect to IL2Rb
	out(8, 2) = r->kfwd * IL2_IL2Ra_IL2Rb; // IL2_IL2Ra_IL2Rb_gc with respect to gc
	out(8, 5) = r->kfwd * gc; // IL2_IL2Ra_IL2Rb_gc with respect to IL2_IL2Ra_IL2Rb
	out(8, 6) = r->kfwd * IL2Rb; // IL2_IL2Ra_IL2Rb_gc with respect to IL2_IL2Ra_gc
	out(8, 7) = r->kfwd * IL2Ra; // IL2_IL2Ra_IL2Rb_gc with respect to IL2_IL2Rb_gc
	out(8, 8) = - r->k8rev - r->k9rev - r->k10rev; // IL2_IL2Ra_IL2Rb_gc with respect to IL2_IL2Ra_IL2Rb_gc
	
	// IL15Ra
	out(9, 9) = -kfbnd * IL15 - r->kfwd * IL15_IL2Rb_gc - r->kfwd * IL15_IL2Rb; // IL15Ra with respect to IL15Ra
	out(9, 10) = k13rev; // IL15Ra with respect to IL15_IL15Ra
	out(9, 11) = - r->kfwd * IL15Ra; // IL15Ra with respect to IL15_IL2Rb
	out(9, 12) = r->k24rev; // IL15Ra with respect to IL15_IL15Ra_IL2Rb
	out(9, 14) = - r->kfwd * IL15Ra; // IL15Ra with respect to IL15_IL2Rb_gc
	out(9, 15) = r->k20rev; // IL15Ra with respect to IL15_IL15Ra_IL2Rb_gc
	
	// IL15_IL15Ra
	out(10, 1) = -r->kfwd * IL15_IL15Ra; // IL15_IL15Ra with respect to IL2Rb
	out(10, 2) = - r->kfwd * IL15_IL15Ra; // IL15_IL15Ra with respect to gc
	out(10, 9) = kfbnd * IL15; // IL15_IL15Ra with respect to IL15Ra
	out(10, 10) = -r->kfwd * IL2Rb - r->kfwd * gc - k13rev; // IL15_IL15Ra with respect to IL15_IL15Ra
	out(10, 12) = r->k23rev; // IL15_IL15Ra with respect to IL15_IL15Ra_IL2Rb
	out(10, 13) = r->k16rev; // IL15_IL15Ra with respect to IL15_IL15Ra_gc
	
	// IL15_IL2Rb
	out(11, 1) = kfbnd * IL15; // IL15_IL2Rb with respect to IL2Rb
	out(11, 2) = -r->kfwd * IL15_IL2Rb; // IL15_IL2Rb with respect to gc
	out(11, 9) = -r->kfwd * IL15_IL2Rb; // IL15_IL2Rb with respect to IL15Ra
	out(11, 11) = -r->kfwd * IL15Ra - r->kfwd * gc - k14rev; // IL15_IL2Rb with respect to IL15_IL2Rb
	out(11, 12) = r->k24rev; // IL15_IL2Rb with respect to IL15_IL15Ra_IL2Rb
	out(11, 14) = r->k17rev; // IL15_IL2Rb with respect to IL15_IL2Rb_gc
	
	// IL15_IL15Ra_IL2Rb
	out(12, 1) = r->kfwd * IL15_IL15Ra; // IL15_IL15Ra_IL2Rb with respect to IL2Rb
	out(12, 2) = -r->kfwd * IL15_IL15Ra_IL2Rb; // IL15_IL15Ra_IL2Rb with respect to gc
	out(12, 9) = r->kfwd * IL15_IL2Rb; // IL15_IL15Ra_IL2Rb with respect to IL15Ra
	out(12, 10) = r->kfwd * IL2Rb; // IL15_IL15Ra_IL2Rb with respect to IL15_IL15Ra
	out(12, 11) = r->kfwd * IL15Ra; // IL15_IL15Ra_IL2Rb with respect to IL15_IL2Rb
	out(12, 12) = -r->kfwd * gc - r->k23rev - r->k24rev; // IL15_IL15Ra_IL2Rb with respect to IL15_IL15Ra_IL2Rb
	out(12, 15) = r->k22rev; // IL15_IL15Ra_IL2Rb with respect to IL15_IL15Ra_IL2Rb_gc
	
	// IL15_IL15Ra_gc
	out(13, 1) = -r->kfwd * IL15_IL15Ra_gc; // IL15_IL15Ra_gc with respect to IL2Rb
	out(13, 2) = r->kfwd * IL15_IL15Ra; // IL15_IL15Ra_gc with respect to gc
	out(13, 10) = r->kfwd * gc; // IL15_IL15Ra_gc with respect to IL15_IL15Ra
	out(13, 13) = -r->kfwd * IL2Rb - r->k16rev; // IL15_IL15Ra_gc with respect to IL15_IL15Ra_gc
	out(13, 15) = r->k21rev; // IL15_IL15Ra_gc with respect to IL15_IL15Ra_IL2Rb_gc
	
	// IL15_IL2Rb_gc
	out(14, 2) = r->kfwd * IL15_IL2Rb; // IL15_IL2Rb_gc with respect to gc
	out(14, 9) = -r->kfwd * IL15_IL2Rb_gc; // IL15_IL2Rb_gc with respect to IL15Ra
	out(14, 11) = r->kfwd * gc; // IL15_IL2Rb_gc with respect to IL15_IL2Rb
	out(14, 14) = -r->kfwd * IL15Ra - r->k17rev; // IL15_IL2Rb_gc with respect to IL15_IL2Rb_gc
	out(14, 15) = r->k20rev; // IL15_IL2Rb_gc with respect to IL15_IL15Ra_IL2Rb_gc
	
	// IL15_IL15Ra_IL2Rb_gc
	out(15, 1) = r->kfwd * IL15_IL15Ra_gc; // IL15_IL15Ra_IL2Rb_gc with respect to IL2Rb
	out(15, 2) = r->kfwd * IL15_IL15Ra_IL2Rb; // IL15_IL15Ra_IL2Rb_gc with respect to gc
	out(15, 9) = r->kfwd * IL15_IL2Rb_gc; // IL15_IL15Ra_IL2Rb_gc with respect to IL15Ra
	out(15, 12) = r->kfwd * gc; // IL15_IL15Ra_IL2Rb_gc with respect to IL15_IL15Ra_IL2Rb
	out(15, 13) = r->kfwd * IL2Rb; // IL15_IL15Ra_IL2Rb_gc with respect to IL15_IL15Ra_gc
	out(15, 14) = r->kfwd * IL15Ra; // IL15_IL15Ra_IL2Rb_gc with respect to IL15_IL2Rb_gc
	out(15, 15) = - r->k20rev - r->k21rev - r->k22rev; // IL15_IL15Ra_IL2Rb_gc with respect to IL15_IL15Ra_IL2Rb_gc
	
	// IL7Ra
	out(16, 16) = -kfbnd * IL7; // IL7Ra with respect to IL7Ra 
	out(16, 17) = k25rev; // IL7Ra with respect to IL7Ra_IL7
	
	// IL7Ra_IL7
	out(17, 2) = - r->kfwd * IL7Ra_IL7; // IL7Ra_IL7 with respect to gc
	out(17, 16) = kfbnd * IL7; // IL7Ra_IL7 with respect to IL7Ra
	out(17, 17) = - k25rev - r->kfwd * gc; // IL7Ra_IL7 with respect to IL7Ra_IL7
	out(17, 18) = r->k27rev; // IL7Ra_IL7 with respect to IL7Ra_gc_IL7
	
	// IL7Ra_gc_IL7
	out(18, 2) = r->kfwd * IL7Ra_IL7; // IL7Ra_gc_IL7 with respect to gc
	out(18, 17) = r->kfwd * gc; // IL7Ra_gc_IL7 with respect to IL7Ra_IL7
	out(18, 18) = - r->k27rev; // IL7Ra_gc_IL7 with respect to IL7Ra_gc_IL7
	
	// IL9R
	out(19, 19) = -kfbnd * IL9; // IL9R with respect to IL9R
	out(19, 20) = k29rev; // IL9R with respect to IL9R_IL9
	
	// IL9R_IL9 
	out(20, 2) = - r->kfwd * IL9R_IL9; // IL9R_IL9 with respect to gc
	out(20, 19) = kfbnd * IL9; // IL9R_IL9 with respect to IL9R
	out(20, 20) = - k29rev - r->kfwd * gc; // IL9R_IL9 with respect to IL9R_IL9
	out(20, 21) = r->k31rev; // IL9R_IL9 with respect to IL9R_gc_IL9
	
	// IL9R_gc_IL9
	out(21, 2) = r->kfwd * IL9R_IL9; // IL9R_gc_IL9 with respect to gc
	out(21, 20) = r->kfwd * gc; // IL9R_gc_IL9 with respect to IL9R_IL9
	out(21, 21) = - r->k31rev; // IL9R_gc_IL9 with respect to IL9R_gc_IL9
}


extern "C" void jacobian_C(double *y_in, double, double *out, double *rxn_in) {
	ratesS r = param(rxn_in);

	jacobian(y_in, &r, out, r.IL2, r.IL15, r.IL7, r.IL9);
}


void fullJacobian(const double * const y, const ratesS * const r, Eigen::Map<JacMat> &out) {
	size_t halfL = activeV.size();
	
	// unless otherwise specified, assume all partial derivatives are 0
	out.setConstant(0.0);

	array <double, 22*22> sub_y;
	jacobian(y, r, sub_y.data(), r->IL2, r->IL15, r->IL7, r->IL9); // jacobian function assigns values to sub_y
	for (size_t ii = 0; ii < halfL; ii++)
		std::copy_n(sub_y.data() + halfL*ii, halfL, out.data() + Nspecies*ii);

	jacobian(y + halfL, r, sub_y.data(), y[44], y[45], y[46], y[47]); // different IL concs for internal case 
	for (size_t ii = 0; ii < halfL; ii++)
		std::copy_n(sub_y.data() + halfL*ii, halfL, out.data() + Nspecies*(ii + halfL) + halfL);

	// Implement trafficking
	double endo = 0;
	double deg = 0;
	double rec = 0;
	for (size_t ii = 0; ii < halfL; ii++) {
		if (activeV[ii]) {
			endo = r->endo + r->activeEndo;
			deg = r->kDeg;
			rec = 0.0;
		} else {
			endo = r->endo;
			deg = r->kDeg*r->sortF;
			rec = r->kRec*(1.0-r->sortF);
		}

		out(ii, ii) = out(ii, ii) - endo; // Endocytosis
		out(ii + halfL, ii + halfL) -= deg + rec; // Degradation
		out(ii + halfL, ii) += endo/internalFrac;
		out(ii, ii + halfL) += rec*internalFrac; // Recycling
	}

	// Ligand degradation
	for (size_t ii = 44; ii < 48; ii++)
		out(ii, ii) -= r->kDeg;

	// Ligand binding
	// Derivative is w.r.t. second number
	const double eIL2 = y[44] / internalV;
	out(44, 44) -= kfbnd * (y[22] + y[23]) / internalV;
	out(22 + 0, 44) = -kfbnd * y[22 + 0]; // IL2 binding to IL2Ra
	out(44, 22) = -kfbnd * eIL2; // IL2 binding to IL2Ra
	out(22 + 1, 44) = -kfbnd * y[22 + 1]; // IL2 binding to IL2Rb
	out(44, 23) = -kfbnd * eIL2; // IL2 binding to IL2Rb
	out(22 + 3, 44) = kfbnd * y[22 + 0]; // IL2 binding to IL2Ra
	out(44, 25) =  k1rev / internalV;
	out(22 + 4, 44) = kfbnd * y[22 + 1]; // IL2 binding to IL2Rb
	out(44, 26) = k2rev / internalV;

	const double eIL15 = y[45] / internalV;
	out(45, 45) -= kfbnd * (y[23] + y[22 + 9]) / internalV;
	out(22 + 1, 45) = -kfbnd * y[22 + 1]; // IL15 binding to IL2Rb
	out(45, 23) = -kfbnd * eIL15; // IL15 binding to IL2Rb
	out(22 + 9, 45) = -kfbnd * y[22 + 9]; // IL15 binding to IL15Ra
	out(45, 31) = -kfbnd * eIL15; // IL15 binding to IL15Ra
	out(22 + 10, 45) =  kfbnd * y[22 + 9]; // IL15 binding to IL15Ra
	out(22 + 11, 45) =  kfbnd * y[22 +  1]; // IL15 binding to IL2Rb
	out(45, 32) = k13rev / internalV;
	out(45, 33) = k14rev / internalV;

	const double eIL7 = y[46] / internalV;
	out(46, 46) -= kfbnd * y[22 + 16] / internalV;
	out(22 + 16, 46) = -kfbnd * y[22 + 16]; // IL7 binding to IL7Ra
	out(46, 22 + 16) = -kfbnd * eIL7; // IL7 binding to IL7Ra
	out(22 + 17, 46) =  kfbnd * y[22 + 16]; // IL7 binding to IL7Ra
	out(46, 39) = k25rev / internalV;

	const double eIL9 = y[47] / internalV;
	out(47, 47) -= kfbnd * y[22 + 19] / internalV;
	out(22 + 19, 47) = -kfbnd * y[22 + 19]; // IL9 binding to IL9R
	out(47, 22 + 19) = -kfbnd * eIL9; // IL9 binding to IL9R
	out(22 + 20, 47) =  kfbnd * y[22 + 19]; // IL9 binding to IL9R
	out(47, 42) = k29rev / internalV; 
}

constexpr bool debugOutput = false;


int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data, N_Vector, N_Vector, N_Vector) {
	ratesS rattes = param(static_cast<double *>(user_data));

	Eigen::Map<JacMat> jac(SM_DATA_D(J));

	// Actually get the Jacobian
	fullJacobian(NV_DATA_S(y), &rattes, jac);

	jac.transposeInPlace();

	if (debugOutput) {
		JacMat A = jac;

		Eigen::JacobiSVD<JacMat> svd(A);
		double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

		if (cond > 1E10)
			std::cout << std::endl << std::endl << jac << std::endl;
	}

	return 0;
}

extern "C" void fullJacobian_C(double *y_in, double, double *dydt, double *rxn_in) {
	ratesS r = param(rxn_in);

	Eigen::Map<JacMat> out(dydt);

	fullJacobian(y_in, &r, out);
}
