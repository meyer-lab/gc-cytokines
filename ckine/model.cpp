#include <algorithm>
#include <cstdio>
#include <numeric>
#include <array>
#include <thread>
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

const array<size_t, 8> recIDX = {{0, 1, 2, 9, 16, 19, 22, 25}};

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


extern "C" void dydt_C(double *y_in, double, double *dydt_out, double *rxn_in) {
	ratesS r(rxn_in);

	dy_dt(y_in, &r, dydt_out, r.ILs.data());
}


void fullModel(const double * const y, const ratesS * const r, double *dydt) {
	// Implement full model.
	fill(dydt, dydt + Nspecies, 0.0);

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


int fullModelCVode (const double, const N_Vector xx, N_Vector dxxdt, void *user_data) {
	ratesS rattes(static_cast<double *>(user_data));

	// Get the data in the right form
	fullModel(NV_DATA_S(xx), &rattes, NV_DATA_S(dxxdt));

	return 0;
}


extern "C" void fullModel_C(const double * const y_in, double, double *dydt_out, double *rxn_in) {
	ratesS r(rxn_in);

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
void solveAutocrineS (const ratesS * const r, N_Vector *y0s, array<double, Nspecies> &y0) {
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
	
	ratesS ratt(sMem->params);
	
	ratt.print();

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


int ewt(N_Vector y, N_Vector w, void *) {
	for (size_t i = 0; i < Nspecies; i++) {
		NV_Ith_S(w, i) = 1.0/(fabs(NV_Ith_S(y, i))*tolIn + tolIn);
	}

	return 0;
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
	
	// Call CVodeWFtolerances to specify the tolerances
	if (CVodeWFtolerances(sMem->cvode_mem, ewt) < 0) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeWFtolerances in solver_setup."));
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


void solver_setup_sensi(solver *sMem, const ratesS * const rr, double *params, array<double, Nspecies> &y0) { 
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
	std::copy_n(params, Nparams, paramArr.begin());
	for(size_t is = 0; is < Nparams; is++) {
		if (paramArr[is] < 0.01) paramArr[is] = 0.01;
	}

	// Specify problem parameter information for sensitivity calculations
	if (CVodeSetSensParams(sMem->cvode_mem, params, paramArr.data(), nullptr) < 0) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeSetSensParams in solver_setup."));
	}
}


void copyOutSensi(double *out, solver *sMem) {
	for (size_t ii = 0; ii < Nparams; ii++) {
		std::copy_n(NV_DATA_S(sMem->yS[ii]), Nspecies, out + ii*Nspecies);
	}
}


extern "C" int runCkineY0 (double *y0in, double *tps, size_t ntps, double *out, double *rxnRatesIn, bool sensi, double *sensiOut) {
	ratesS rattes(rxnRatesIn);
	size_t itps = 0;

	array<double, Nspecies> y0;
	std::copy_n(y0in, y0.size(), y0.begin());

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


extern "C" int runCkine (double *tps, size_t ntps, double *out, double *rxnRatesIn, bool sensi, double *sensiOut) {
	ratesS rattes(rxnRatesIn);

	array<double, Nspecies> y0 = solveAutocrine(&rattes);

	return runCkineY0 (y0.data(), tps, ntps, out, rxnRatesIn, sensi, sensiOut);
}


extern "C" int runCkineParallel (double *rxnRatesIn, double tp, size_t nDoses, bool sensi, double *out, double *sensiOut) {
	vector<int> retVals(nDoses, -1);
	vector<std::thread> ts;

	// Make a task that handles all the refs
	auto lamTask = [&tp, &out, &rxnRatesIn, &sensi, &sensiOut](size_t ii, int *retVal) {
		*retVal = runCkine (&tp, 1, out + Nspecies*ii, rxnRatesIn + ii*Nparams, sensi, sensiOut + Nspecies*Nparams*ii);
	};

	// Actually run the simulations
	for (size_t ii = 0; ii < nDoses; ii++)
		ts.push_back(std::thread(lamTask, ii, retVals.data() + ii));

	// Synchronize all threads
	for (auto& th:ts) th.join();

	// Get the worst case to return
	return *std::min_element(retVals.begin(), retVals.end());
}


void jacobian(const double * const y, const ratesS * const r, double * const dydt, const double * const ILs) {
	// IL2 in nM
	const double IL2Ra = y[0];
	const double IL2Rb = y[1];
	const double gc = y[2];
	const double IL2_IL2Ra = y[3];
	const double IL2_IL2Rb = y[4];
	const double IL2_IL2Ra_gc = y[6];
	const double IL2_IL2Rb_gc = y[7];
	
	// IL15 in nM
	const double IL15Ra = y[9];
	const double IL15_IL15Ra = y[10];
	const double IL15_IL2Rb = y[11];
	const double IL15_IL15Ra_gc = y[13];
	const double IL15_IL2Rb_gc = y[14];

	Eigen::Map<Eigen::Matrix<double, halfL, halfL, Eigen::RowMajor>> out(dydt);
	
	// unless otherwise specified, assume all partial derivatives are 0
	out.setConstant(0.0);

	auto complexCkine = [&out, &gc, &r, &y](const size_t ij, const size_t RaIDX, const double revOne, const double revTwo) {
		out(RaIDX, ij+1) = -r->kfwd * y[RaIDX]; // Ra with respect to IL_Rb
		out(RaIDX, ij+4) = -r->kfwd * y[RaIDX]; // Ra with respect to IL_Rb_gc

		out(2, 2) -= r->kfwd * (y[ij] + y[ij+1] + y[ij+2]); // gc with respect to gc
		out(2, ij) = - r->kfwd * gc; // gc with respect to IL_Ra
		out(2, ij+1) = - r->kfwd * gc; // gc with respect to IL_Rb
		out(2, ij+2) = - r->kfwd * gc; // gc with respect to IL_Ra_Rb

		out(1, ij) = - r->kfwd * y[1]; // Rb with respect to IL_Ra
		out(1, ij+3) = - r->kfwd * y[1]; // Rb with respect to IL_Ra_gc

		out(ij, 1) = -r->kfwd * y[ij]; // IL_Ra with respect to Rb
		out(ij, 2) = - r->kfwd * y[ij]; // IL_Ra with respect to gc

		out(ij+5, ij+2) = r->kfwd * gc; // IL_Ra_Rb_gc with respect to IL_Ra_Rb

		out(ij+4, 2) = r->kfwd * y[ij+1]; // IL_Rb_gc with respect to gc
		out(ij+4, RaIDX) = -r->kfwd * y[ij+4]; // IL_Rb_gc with respect to Ra
		out(ij+4, ij+1) = r->kfwd * gc; // IL_Rb_gc with respect to IL_Rb
		out(ij+4, ij+4) = -r->kfwd * y[RaIDX] - revOne; // IL_Rb_gc with respect to IL_Rb_gc
		out(ij+4, ij+5) = revTwo; // IL_Rb_gc with respect to IL_Ra_Rb_gc

		out(2, ij+4) = revOne; // gc with respect to IL_Rb_gc
		out(ij+1, ij+4) = revOne; // IL_Rb with respect to IL_Rb_gc
		out(RaIDX, ij+5) = revTwo; // Ra with respect to IL_Ra_Rb_gc

		out(ij+5, RaIDX) = r->kfwd * y[ij+4]; // IL_Ra_Rb_gc with respect to Ra
		out(ij+5, 1) = r->kfwd * y[ij+3]; // IL_Ra_Rb_gc with respect to Rb
		out(ij+5, 2) = r->kfwd * y[ij+2]; // IL_Ra_Rb_gc with respect to gc
		out(ij+5, ij+3) = r->kfwd * y[1]; // IL_Ra_Rb_gc with respect to IL_Ra_gc
		out(ij+5, ij+4) = r->kfwd * y[RaIDX]; // IL_Ra_Rb_gc with respect to IL_Rb_gc
		out(ij+2, RaIDX) = r->kfwd * y[ij+1]; // IL_Ra_Rb with respect to Ra
		out(ij+2, 1) = r->kfwd * y[ij]; // IL_Ra_Rb with respect to Rb
		out(ij+2, 2) = -r->kfwd * y[ij+2]; // IL_Ra_Rb with respect to gc
		out(ij+2, ij) = r->kfwd * y[1]; // IL_Ra_Rb with respect to IL_Ra
		out(ij+2, ij+1) = r->kfwd * y[RaIDX]; // IL_Ra_Rb with respect to IL_Rb

		out(ij+1, RaIDX) = -r->kfwd * y[ij+1]; // IL2_IL2Rb with respect to IL2Ra
		out(ij+1, 2) = - r->kfwd * y[ij+1]; // IL2_IL2Rb with respect to gc
		out(ij+3, 1) = -r->kfwd * y[ij+3]; // IL2_IL2Ra_gc with respect to IL2Rb
		out(ij+3, 2) = r->kfwd * y[ij]; // IL2_IL2Ra_gc with respect to gc
		out(ij+3, ij) = r->kfwd * gc; // IL2_IL2Ra_gc with respect to IL2_IL2Ra
	};

	complexCkine(3, 0, r->k5rev, r->k8rev); // IL2
	complexCkine(10, 9, r->k17rev, r->k20rev); // IL15
	
	// IL2Ra
	out(0, 0) = -kfbnd * ILs[0] - r->kfwd * IL2_IL2Rb_gc - r->kfwd * IL2_IL2Rb; // IL2Ra with respect to IL2Ra
	out(0, 3) = k1rev; // IL2Ra with respect to IL2_IL2Ra
	out(0, 5) = r->k12rev; // IL2Ra with respect to IL2_IL2Ra_IL2Rb
	
	// IL2Rb
	out(1, 1) = -kfbnd * (ILs[0] + ILs[1]) - r->kfwd * (IL2_IL2Ra_gc + IL2_IL2Ra + IL15_IL15Ra_gc + IL15_IL15Ra); // partial derivative of IL2Rb with respect to IL2Rb
	out(1, 4) = k2rev; // IL2Rb with respect to IL2_IL2Rb
	out(1, 5) = r->k11rev; // IL2Rb with respect to IL2_IL2Ra_IL2Rb
	out(1, 8) = r->k9rev; // IL2Rb with respect to IL2_IL2Ra_IL2Rb_gc
	out(1, 11) = k14rev; // IL2Rb with respect to IL15_IL2Rb
	out(1, 12) = r->k23rev; // IL2Rb with respect to IL15_IL15Ra_IL2Rb
	out(1, 15) = r->k21rev; // IL2Rb with respect to IL15_IL15Ra_IL2Rb_gc
	
	// gc
	out(2, 6) = r->k4rev; // gc with respect to IL2_IL2Ra_gc
	out(2, 8) = r->k10rev; // gc with respect to IL2_IL2Ra_IL2Rb_gc
	out(2, 13) = r->k16rev; // gc with respect to IL15_IL15Ra_gc
	out(2, 15) = r->k22rev; // gc with respect to IL15_IL15Ra_IL2Rb_gc
	
	// IL2_IL2Ra
	out(3, 0) = kfbnd * ILs[0]; // IL2_IL2Ra with respect to IL2Ra
	out(3, 3) = -r->kfwd * IL2Rb - r->kfwd * gc - k1rev; // IL2_IL2Ra with respect to IL2_IL2Ra
	out(3, 5) = r->k11rev; // IL2_IL2Ra with respect to IL2_IL2Ra_IL2Rb
	out(3, 6) = r->k4rev; // IL2_IL2Ra with respect to IL2_IL2Ra_gc
	
	// IL2_IL2Rb
	out(4, 1) = kfbnd * ILs[0]; // IL2_IL2Rb with respect to IL2Rb
	out(4, 4) = -r->kfwd * IL2Ra - r->kfwd * gc - k2rev; // IL2_IL2Rb with respect to IL2_IL2Rb
	out(4, 5) = r->k12rev; // IL2_IL2Rb with respect to IL2_IL2Ra_IL2Rb
	
	// IL2_IL2Ra_IL2Rb
	out(5, 5) = -r->kfwd * gc - r->k11rev - r->k12rev; // IL2_IL2Ra_IL2Rb with respect to IL2_IL2Ra_IL2Rb
	out(5, 8) = r->k10rev; // IL2_IL2Ra_IL2Rb with respect to IL2_IL2Ra_IL2Rb_gc
	
	// IL2_IL2Ra_gc
	out(6, 6) = -r->kfwd * IL2Rb - r->k4rev; // IL2_IL2Ra_gc with respect to IL2_IL2Ra_gc
	out(6, 8) = r->k9rev; // IL2_IL2Ra_gc with respect to IL2_IL2Ra_IL2Rb_gc
	
	// IL2_IL2Ra_IL2Rb_gc
	out(8, 8) = - r->k8rev - r->k9rev - r->k10rev; // IL2_IL2Ra_IL2Rb_gc with respect to IL2_IL2Ra_IL2Rb_gc
	
	// IL15Ra
	out(9, 9) = -kfbnd * ILs[1] - r->kfwd * IL15_IL2Rb_gc - r->kfwd * IL15_IL2Rb; // IL15Ra with respect to IL15Ra
	out(9, 10) = k13rev; // IL15Ra with respect to IL15_IL15Ra
	out(9, 12) = r->k24rev; // IL15Ra with respect to IL15_IL15Ra_IL2Rb
	
	// IL15_IL15Ra
	out(10, 9) = kfbnd * ILs[1]; // IL15_IL15Ra with respect to IL15Ra
	out(10, 10) = -r->kfwd * IL2Rb - r->kfwd * gc - k13rev; // IL15_IL15Ra with respect to IL15_IL15Ra
	out(10, 12) = r->k23rev; // IL15_IL15Ra with respect to IL15_IL15Ra_IL2Rb
	out(10, 13) = r->k16rev; // IL15_IL15Ra with respect to IL15_IL15Ra_gc
	
	// IL15_IL2Rb
	out(11, 1) = kfbnd * ILs[1]; // IL15_IL2Rb with respect to IL2Rb
	out(11, 11) = -r->kfwd * IL15Ra - r->kfwd * gc - k14rev; // IL15_IL2Rb with respect to IL15_IL2Rb
	out(11, 12) = r->k24rev; // IL15_IL2Rb with respect to IL15_IL15Ra_IL2Rb
	
	// IL15_IL15Ra_IL2Rb
	out(12, 12) = -r->kfwd * gc - r->k23rev - r->k24rev; // IL15_IL15Ra_IL2Rb with respect to IL15_IL15Ra_IL2Rb
	out(12, 15) = r->k22rev; // IL15_IL15Ra_IL2Rb with respect to IL15_IL15Ra_IL2Rb_gc
	
	// IL15_IL15Ra_gc
	out(13, 13) = -r->kfwd * IL2Rb - r->k16rev; // IL15_IL15Ra_gc with respect to IL15_IL15Ra_gc
	out(13, 15) = r->k21rev; // IL15_IL15Ra_gc with respect to IL15_IL15Ra_IL2Rb_gc
	
	// IL15_IL15Ra_IL2Rb_gc
	out(15, 15) = - r->k20rev - r->k21rev - r->k22rev; // IL15_IL15Ra_IL2Rb_gc with respect to IL15_IL15Ra_IL2Rb_gc


	auto simpleCkine = [&out, &gc, &r, &y](const size_t ij, const double revOne, const double revTwo, const double IL) {
		out(2, 2) -= r->kfwd * y[ij+1]; // gc with respect to gc
		out(2, ij+1) = -r->kfwd * gc; // gc with respect to Ra_IL
		out(2, ij+2) = revTwo; // gc with respect to Ra_gc_IL

		out(ij, ij  ) = -kfbnd * IL; // Ra with respect to Ra
		out(ij, ij+1) = revOne; // IL_Ra

		out(ij+1, 2) = - r->kfwd * y[ij+1]; // IL_Ra with respect to gc
		out(ij+1, ij) = kfbnd * IL; // IL_Ra with respect to Ra
		out(ij+1, ij+1) = - revOne - r->kfwd * gc; // IL_Ra with respect to IL_Ra
		out(ij+1, ij+2) = revTwo; // IL_Ra with respect to IL_Ra_gc

		out(ij+2,    2) = r->kfwd * y[ij+1]; // IL_Ra_gc with respect to gc
		out(ij+2, ij+1) = r->kfwd * gc; // IL_Ra_gc with respect to IL_Ra
		out(ij+2, ij+2) = -revTwo; // IL_Ra_gc with respect to IL_Ra_gc
	};

	simpleCkine(16, k25rev, r->k27rev, ILs[2]); // IL7
	simpleCkine(19, k29rev, r->k31rev, ILs[3]); // IL9
	simpleCkine(22, k32rev, r->k33rev, ILs[4]); // IL4
	simpleCkine(25, k34rev, r->k35rev, ILs[5]); // IL21
}


extern "C" void jacobian_C(double *y_in, double, double *out, double *rxn_in) {
	ratesS r(rxn_in);

	jacobian(y_in, &r, out, r.ILs.data());
}


void fullJacobian(const double * const y, const ratesS * const r, Eigen::Map<JacMat> &out) {
	
	// unless otherwise specified, assume all partial derivatives are 0
	out.setConstant(0.0);

	array <double, (halfL*halfL)> sub_y;
	jacobian(y, r, sub_y.data(), r->ILs.data()); // jacobian function assigns values to sub_y
	for (size_t ii = 0; ii < halfL; ii++)
		std::copy_n(sub_y.data() + halfL*ii, halfL, out.data() + Nspecies*ii);

	jacobian(y + halfL, r, sub_y.data(), y + halfL*2); // different IL concs for internal case 
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
	for (size_t ii = (halfL*2); ii < ((halfL*2)+6); ii++)
		out(ii, ii) -= r->kDeg;

	// Ligand binding
	// Derivative is w.r.t. second number
	const double eIL2 = y[56] / internalV;
	out(56, 56) -= kfbnd * (y[halfL] + y[halfL+1]) / internalV;
	out(halfL + 0, 56) = -kfbnd * y[halfL + 0]; // IL2 binding to IL2Ra
	out(56, halfL) = -kfbnd * eIL2; // IL2 binding to IL2Ra
	out(halfL + 1, 56) = -kfbnd * y[halfL + 1]; // IL2 binding to IL2Rb
	out(56, halfL+1) = -kfbnd * eIL2; // IL2 binding to IL2Rb
	out(halfL + 3, 56) = kfbnd * y[halfL + 0]; // IL2 binding to IL2Ra
	out(56, halfL+3) =  k1rev / internalV;
	out(halfL + 4, 56) = kfbnd * y[halfL + 1]; // IL2 binding to IL2Rb
	out(56, halfL+4) = k2rev / internalV;

	const double eIL15 = y[57] / internalV;
	out(57, 57) -= kfbnd * (y[halfL+1] + y[halfL + 9]) / internalV;
	out(halfL + 1, 57) = -kfbnd * y[halfL + 1]; // IL15 binding to IL2Rb
	out(57, halfL+1) = -kfbnd * eIL15; // IL15 binding to IL2Rb
	out(halfL + 9, 57) = -kfbnd * y[halfL + 9]; // IL15 binding to IL15Ra
	out(57, halfL+9) = -kfbnd * eIL15; // IL15 binding to IL15Ra
	out(halfL + 10, 57) =  kfbnd * y[halfL + 9]; // IL15 binding to IL15Ra
	out(halfL + 11, 57) =  kfbnd * y[halfL +  1]; // IL15 binding to IL2Rb
	out(57, halfL+10) = k13rev / internalV;
	out(57, halfL+11) = k14rev / internalV;

	auto simpleCkine = [&](const size_t ij, const size_t ix, const double revRate) {
		const double eIL = y[ix] / internalV;
		out(ix, ix) -= kfbnd * y[halfL + ij] / internalV;
		out(halfL + ij, ix) = -kfbnd * y[halfL + ij];
		out(ix, halfL + ij) = -kfbnd * eIL;
		out(halfL + ij + 1, ix) =  kfbnd * y[halfL + ij];
		out(ix, halfL + ij + 1) = revRate / internalV;
	};

	simpleCkine(16, 58, k25rev); // IL7
	simpleCkine(19, 59, k29rev); // IL9
	simpleCkine(22, 60, k32rev); // IL4
	simpleCkine(25, 61, k34rev); // IL21
}


int Jac(realtype, N_Vector y, N_Vector, SUNMatrix J, void *user_data, N_Vector, N_Vector, N_Vector) {
	ratesS rattes(static_cast<double *>(user_data));

	Eigen::Map<JacMat> jac(SM_DATA_D(J));

	// Actually get the Jacobian
	fullJacobian(NV_DATA_S(y), &rattes, jac);

	jac.transposeInPlace();

	return 0;
}

extern "C" void fullJacobian_C(double *y_in, double, double *dydt, double *rxn_in) {
	ratesS r(rxn_in);

	Eigen::Map<JacMat> out(dydt);

	fullJacobian(y_in, &r, out);
}
