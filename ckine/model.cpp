#include <algorithm>
#include <cstdio>
#include <numeric>
#include <array>
#include <thread>
#include <mutex>
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
#include "reaction.hpp"
#include "jacobian.hpp"

using std::array;
using std::copy;
using std::vector;
using std::fill;
using std::string;
using std::endl;
using std::cout;

std::mutex print_mutex; // mutex to prevent threads printing on top of each other

extern "C" void dydt_C(double *y_in, double, double *dydt_out, double *rxn_in) {
	ratesS r(rxn_in);

	dy_dt(y_in, &r, dydt_out, r.ILs.data());
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
void solveAutocrineS (const ratesS * const r, N_Vector *y0s) {
	array<double, Nspecies> y0 = solveAutocrine(r);

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
	array<double, Nparams> params;
};


static void errorHandler(int error_code, const char *module, const char *function, char *msg, void *ehdata) {
	if (error_code == CV_WARNING) return;
	solver *sMem = static_cast<solver *>(ehdata);
	ratesS ratt(sMem->params.data());

	std::lock_guard<std::mutex> lock(print_mutex);

	std::cout << "Internal CVode error in " << function << ", module: " << module << ", error code: " << error_code << std::endl;
	std::cout << msg << std::endl;
	std::cout << "Parameters: ";

	for (size_t ii = 0; ii < Nparams; ii++) {
		std::cout << sMem->params[ii] << "\t";
	}
	
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


int Jac(realtype, N_Vector y, N_Vector, SUNMatrix J, void *user_data, N_Vector, N_Vector, N_Vector) {
	ratesS rattes(static_cast<double *>(user_data));

	Eigen::Map<JacMat> jac(SM_DATA_D(J));

	// Actually get the Jacobian
	fullJacobian(NV_DATA_S(y), &rattes, jac);

	jac.transposeInPlace();

	return 0;
}


void solver_setup(solver *sMem, const double * const params) {
	std::copy_n(params, Nparams, sMem->params.begin());

	/* Call CVodeCreate to create the solver memory and specify the
	 * Backward Differentiation Formula and the use of a Newton iteration */
	sMem->cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
	if (sMem->cvode_mem == nullptr) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeCreate in solver_setup."));
	}
	
	CVodeSetErrHandlerFn(sMem->cvode_mem, &errorHandler, static_cast<void *>(sMem));

	// Pass along the parameter structure to the differential equations
	if (CVodeSetUserData(sMem->cvode_mem, static_cast<void *>(sMem->params.data())) < 0) {
		solverFree(sMem);
		throw std::runtime_error(string("Error calling CVodeSetUserData in solver_setup."));
	}

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

	CVodeSetMaxNumSteps(sMem->cvode_mem, 800000);

	// Now we are doing a sensitivity analysis
	if (sMem->sensi) {
		// Set sensitivity initial conditions
		sMem->yS = N_VCloneVectorArray(Nparams, sMem->state);
		ratesS rattes(sMem->params.data());
		solveAutocrineS(&rattes, sMem->yS);

		// Call CVodeSensInit1 to activate forward sensitivity computations
		// and allocate internal memory for CVODES related to sensitivity
		// calculations. Computes the right-hand sides of the sensitivity
		// ODE, one at a time
		if (CVodeSensInit(sMem->cvode_mem, Nparams, CV_SIMULTANEOUS, nullptr, sMem->yS) < 0) {
			solverFree(sMem);
			throw std::runtime_error(string("Error calling CVodeSensInit in solver_setup."));
		}

		// Call CVodeSensEEtolerances to estimate tolerances for sensitivity 
		// variables based on the rolerances supplied for states variables and 
		// the scaling factor pbar
		if (CVodeSensEEtolerances(sMem->cvode_mem) < 0) {
			solverFree(sMem);
			throw std::runtime_error(string("Error calling CVodeSensSStolerances in solver_setup."));
		}

		array<double, Nparams> paramArr;
		std::copy_n(params, Nparams, paramArr.begin());
		for(size_t is = 0; is < Nparams; is++) {
			if (paramArr[is] < 0.01) paramArr[is] = 0.01;
		}

		// Specify problem parameter information for sensitivity calculations
		if (CVodeSetSensParams(sMem->cvode_mem, sMem->params.data(), paramArr.data(), nullptr) < 0) {
			solverFree(sMem);
			throw std::runtime_error(string("Error calling CVodeSetSensParams in solver_setup."));
		}
	}
}


void copyOutSensi(double *out, solver *sMem) {
	for (size_t ii = 0; ii < Nparams; ii++) {
		std::copy_n(NV_DATA_S(sMem->yS[ii]), Nspecies, out + ii*Nspecies);
	}
}


extern "C" int runCkineY0 (const double * const y0in, double * const tps, const size_t ntps, double * const out, const double * const rxnRatesIn, const bool sensi, double * const sensiOut) {
	ratesS rattes(rxnRatesIn);
	size_t itps = 0;

	solver sMem;
	sMem.sensi = sensi;

	// Just the full model
	sMem.state = N_VNew_Serial(static_cast<long>(Nspecies));
	std::copy_n(y0in, Nspecies, NV_DATA_S(sMem.state));

	solver_setup(&sMem, rxnRatesIn);

	double tret = 0;

	if (tps[0] < std::numeric_limits<double>::epsilon()) {
		std::copy_n(y0in, Nspecies, out);

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
		std::copy_n(NV_DATA_S(sMem.state), Nspecies, out + Nspecies*itps);

		if (sensi) {
			CVodeGetSens(sMem.cvode_mem, &tps[itps], sMem.yS);
			copyOutSensi(sensiOut + Nspecies*Nparams*itps, &sMem);
		}
	}

	solverFree(&sMem);
	return 0;
}


extern "C" int runCkine (double * const tps, const size_t ntps, double * const out, const double * const rxnRatesIn, const bool sensi, double * const sensiOut) {
	ratesS rattes(rxnRatesIn);

	array<double, Nspecies> y0 = solveAutocrine(&rattes);

	return runCkineY0 (y0.data(), tps, ntps, out, rxnRatesIn, sensi, sensiOut);
}


extern "C" int runCkineParallel (const double * const rxnRatesIn, double tp, size_t nDoses, bool sensi, double *out, double *sensiOut) {
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


extern "C" void fullJacobian_C(double *y_in, double, double *dydt, double *rxn_in) {
	ratesS r(rxn_in);

	Eigen::Map<JacMat> out(dydt);

	fullJacobian(y_in, &r, out);
}
