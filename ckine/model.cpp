#define ADEPT_STORAGE_THREAD_SAFE

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <array>
#include <thread>
#include <vector>
#include <list>
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
#include <adept.h>
#include "reaction.hpp"
#include "thread_pool.hpp"

using std::array;
using std::copy;
using std::vector;
using std::fill;
using std::string;
using std::endl;
using std::cout;
using adept::adouble;

constexpr double solveTol = 1.0E-3;

static void errorHandler(int, const char *, const char *, char *, void *);
int Jac(double, N_Vector, N_Vector, SUNMatrix, void *, N_Vector, N_Vector, N_Vector);
int JacB(double, N_Vector, N_Vector, N_Vector, SUNMatrix, void *, N_Vector, N_Vector, N_Vector);
int fullModelCVode (const double, const N_Vector, N_Vector, void *);
static int fB(double, N_Vector y, N_Vector yB, N_Vector yBdot, void *user_dataB);
static int fQB(double, N_Vector y, N_Vector yB, N_Vector qBdot, void *user_dataB);

std::mutex print_mutex; // mutex to prevent threads printing on top of each other

typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> eigenV;
typedef Eigen::Map<Eigen::Matrix<double, Nspecies, 1>> eigenVC;
typedef Eigen::Matrix<double, Nspecies, Eigen::Dynamic> x0JacM;


extern "C" void dydt_C(double *y_in, double, double *dydt_out, double *rxn_in) {
	std::vector<double> v(rxn_in, rxn_in + Nparams);
	ratesS<double> r(v);

	dy_dt(y_in, &r.surface, dydt_out, r.ILs.data());
}


extern "C" void fullModel_C(const double * const y_in, double, double *dydt_out, double *rxn_in) {
	std::vector<double> v(rxn_in, rxn_in + Nparams);
	ratesS<double> r(v);

	fullModel(y_in, &r, dydt_out);
}

constexpr bool debug = false;


class solver {
public:
	void *cvode_mem;
	SUNLinearSolver LS, LSB;
	N_Vector state, qB, yB;
	SUNMatrix A, AB;
	bool sensi;
	int ncheck, indexB;
	double tret;
	vector<double> params;
	array<double, Nspecies> activities;
	adept::Stack stack;
	double preT;
	const double *preL;

	void commonSetup(vector<double> paramsIn, const double preTin, const double * const preLin) {
		tret = 0.0;
		params = paramsIn;
		preT = preTin;
		preL = preLin;

		// Setup state variable by solving for autocrine
		ratesS<double> rattes(params);
		array<double, Nspecies> y0 = solveAutocrine(&rattes);
		state = N_VNew_Serial(static_cast<long>(Nspecies));
		std::copy_n(y0.data(), Nspecies, NV_DATA_S(state));

		/* Call CVodeCreate to create the solver memory and specify the
		 * Backward Differentiation Formula and the use of a Newton iteration */
		cvode_mem = CVodeCreate(CV_BDF);
		if (cvode_mem == nullptr) {
			throw std::runtime_error(string("Error calling CVodeCreate in solver_setup."));
		}
		
		CVodeSetErrHandlerFn(cvode_mem, &errorHandler, static_cast<void *>(this));

		// Pass along the parameter structure to the differential equations
		if (CVodeSetUserData(cvode_mem, static_cast<void *>(this)) < 0) {
			throw std::runtime_error(string("Error calling CVodeSetUserData in solver_setup."));
		}

		// Initialize the integrator rhs function in y'=f(t,y), the inital T0, & the initial dependent y
		if (CVodeInit(cvode_mem, fullModelCVode, 0.0, state) < 0) {
			throw std::runtime_error(string("Error calling CVodeInit in solver_setup."));
		}
		
		// Set the scalar relative and absolute tolerances
		if (CVodeSStolerances(cvode_mem, 1.0E-9, 1.0E-9) < 0) {
			throw std::runtime_error(string("Error calling CVodeSStolerances in solver_setup."));
		}

		A = SUNDenseMatrix(Nspecies, Nspecies);
		LS = SUNDenseLinearSolver(state, A);
		
		// Call CVDense to specify the CVDENSE dense linear solver
		if (CVDlsSetLinearSolver(cvode_mem, LS, A) < 0) {
			throw std::runtime_error(string("Error calling CVDlsSetLinearSolver in solver_setup."));
		}

		CVDlsSetJacFn(cvode_mem, Jac);

		CVodeSetMaxNumSteps(cvode_mem, 80000);

		// Call CVodeSetConstraints to initialize constraints
		N_Vector constraints = N_VNew_Serial(static_cast<long>(Nspecies));
		N_VConst(1.0, constraints); // all 1's for nonnegative solution values
		if (CVodeSetConstraints(cvode_mem, constraints) < 0) {
			throw std::runtime_error(string("Error calling CVodeSetConstraints in solver_setup."));
		}
		N_VDestroy(constraints);
	}


	solver(vector<double> paramsIn, const double preTin, const double * const preLin) {
		sensi = false;
		commonSetup(paramsIn, preTin, preLin);
	}

	solver(vector<double> paramsIn, array<double, Nspecies> actIn, const double preTin, const double * const preLin) {
		sensi = true;
		std::copy(actIn.begin(), actIn.end(), activities.begin());
		commonSetup(paramsIn, preTin, preLin);

		// CVodeAdjInit to update CVODES memory block by allocting the internal memory needed for backward integration
		constexpr int steps = 10; // no. of integration steps between two consecutive ckeckpoints
		if (CVodeAdjInit(cvode_mem, steps, CV_HERMITE) < 0) {
			throw std::runtime_error(string("Error calling CVodeAdjInit in solver_setup."));
		}
	}


	void backward (double TB1) {
		indexB = 1;
		yB = N_VNew_Serial(Nspecies); // Initialize yB
		qB = N_VNew_Serial(params.size()); // Initialize qB
		std::copy_n(activities.begin(), Nspecies, NV_DATA_S(yB));
		N_VConst(0.0, qB);

		// CVodeCreateB to specify the solution method for the backward problem
		if (CVodeCreateB(cvode_mem, CV_BDF, &indexB) < 0)
			throw std::runtime_error(string("Error calling CVodeCreateB in solver_setup."));

		// Call CVodeInitB to allocate internal memory and initialize the backward problem
		if (CVodeInitB(cvode_mem, indexB, fB, TB1, yB) < 0)
			throw std::runtime_error(string("Error calling CVodeInitB in solver_setup."));

		// Set the scalar relative and absolute tolerances
		if (CVodeSStolerancesB(cvode_mem, indexB, solveTol, solveTol) < 0)
			throw std::runtime_error(string("Error calling CVodeSStolerancesB in solver_setup."));

		// Attach the user data for backward problem
		if (CVodeSetUserDataB(cvode_mem, indexB, static_cast<void *>(this)) < 0)
			throw std::runtime_error(string("Error calling CVodeSetUserDataB in solver_setup."));

		AB = SUNDenseMatrix(Nspecies, Nspecies);
		LSB = SUNDenseLinearSolver(state, AB);
		
		// Call CVDense to specify the CVDENSE dense linear solver
		if (CVodeSetLinearSolverB(cvode_mem, indexB, LSB, AB) < 0) {
			throw std::runtime_error(string("Error calling CVodeSetLinearSolverB in solver_setup."));
		}

		// Set the user-supplied Jacobian routine JacB
		if (CVodeSetJacFnB(cvode_mem, indexB, JacB) < 0) {
			throw std::runtime_error(string("Error calling CVodeSetJacFnB in solver_setup."));
		}

		// Allocate internal memory and initialize backward quadrature integration
		if (CVodeQuadInitB(cvode_mem, indexB, fQB, qB) < 0) {
			throw std::runtime_error(string("Error calling CVodeQuadInitB in solver_setup."));
		}

		// Whether or not the quadrature variables are to be used in the step size control
		if (CVodeSetQuadErrConB(cvode_mem, indexB, true) < 0) {
			throw std::runtime_error(string("Error calling CVodeSetQuadErrConB in solver_setup."));
		}

		// Specify the scalar relative and absolute tolerances for the backward problem
		if (CVodeQuadSStolerancesB(cvode_mem, indexB, solveTol, solveTol) < 0) {
			throw std::runtime_error(string("Error calling CVodeQuadSStolerancesB in solver_setup."));
		}

		CVodeSetMaxNumStepsB(cvode_mem, indexB, 80000);
	}

	int CVodeRun(const double endT) {
		int returnVal;

		if (endT < tret) {
			cout << "Can't go backwards in forward pass." << std::endl;
			return -1;
		}

		if (sensi) {
			returnVal = CVodeF(cvode_mem, endT, state, &tret, CV_NORMAL, &ncheck);
		} else {
			returnVal = CVode(cvode_mem, endT, state, &tret, CV_NORMAL);
		}

		if (returnVal >= 0 && debug) {
			long nst;
			CVodeGetNumSteps(cvode_mem, &nst);
			cout << "Number of steps: " << nst << std::endl;
			cout << "Final time: " << tret << std::endl;
		}
		
		if (returnVal < 0) cout << "CVode error in CVode. Code: " << returnVal << std::endl;

		return returnVal;
	}

	ratesS<double> getRates() {
		return ratesS<double>(params);
	}

	double getActivity() {
		return std::inner_product(activities.begin(), activities.end(), NV_DATA_S(state), 0.0);
	}

	int getAdjSens(const double t0B, double *Sout, x0JacM &x0p) {
		Eigen::Map<Eigen::Matrix<double, Nspecies, 1>> St0(NV_DATA_S(yB), Nspecies);
		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> Sqt0(NV_DATA_S(qB), params.size());
		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> SoutV(Sout, params.size());
		std::copy_n(activities.begin(), Nspecies, NV_DATA_S(yB));
		N_VConst(0.0, qB);

		if (t0B > std::numeric_limits<double>::epsilon()) {
			if (CVodeReInitB(cvode_mem, indexB, t0B, yB) < 0)
				cout << "CVodeReInitB error at 0." << std::endl;

			if (CVodeQuadReInitB(cvode_mem, indexB, qB) < 0)
				cout << "CVodeQuadReInitB error at 0." << std::endl;

			if (CVodeB(cvode_mem, 0.0, CV_NORMAL) < 0)
				cout << "CVodeB error at 0, while integrating back from " << t0B << "." << std::endl;

			CVodeGetB(cvode_mem, indexB, &tret, yB);
			CVodeGetQuadB(cvode_mem, indexB, &tret, qB);
		}

		SoutV = St0.transpose()*x0p - Sqt0.transpose();

		return 0;
	}

	~solver() {
		if (sensi) {
			CVodeSensFree(cvode_mem);
			N_VDestroy_Serial(qB);
			N_VDestroy_Serial(yB);
			SUNLinSolFree(LSB);
			SUNMatDestroy(AB);
		}

		N_VDestroy_Serial(state);
		CVodeFree(&cvode_mem);
		SUNLinSolFree(LS);
		SUNMatDestroy(A);
	}
};

void preTreat(double tDiff, std::array<double, Nlig> &ILs, const double *preL) {
	tDiff = fabs(60*tDiff); // Scale this to be on the order of seconds

	for (size_t ii = 0; ii < Nlig; ii++) {
		ILs[ii] = preL[ii] + (ILs[ii] - preL[ii])*exp(-tDiff);
	}
}

// fB routine. Compute fB(t,y,yB). 
static int fB(double t, N_Vector y, N_Vector yB, N_Vector yBdot, void *user_data) {
	solver *sMem = static_cast<solver *>(user_data);
	ratesS<double> rattes = sMem->getRates();

	if (t < sMem->preT)
		preTreat(t - sMem->preT, rattes.ILs, sMem->preL);

	std::array<adept::adouble, Nspecies> ya, dydt;

	adept::set_values(&ya[0], Nspecies, NV_DATA_S(y));

	sMem->stack.new_recording();

	// Get the data in the right form
	fullModel(ya.data(), &rattes, dydt.data());

	adouble yOut = 0;
	yOut = -std::inner_product(dydt.begin(), dydt.end(), NV_DATA_S(yB), yOut);

	sMem->stack.independent(&ya[0], Nspecies);
	sMem->stack.dependent(&yOut, 1);

	sMem->stack.jacobian(NV_DATA_S(yBdot));

	return 0;
}


int Jac(double t, N_Vector yv, N_Vector, SUNMatrix J, void *user_data, N_Vector, N_Vector, N_Vector) {
	solver *sMem = static_cast<solver *>(user_data);
	ratesS<double> rattes = sMem->getRates();

	if (t < sMem->preT)
		preTreat(t - sMem->preT, rattes.ILs, sMem->preL);

	Eigen::Map<Eigen::Matrix<double, Nspecies, Nspecies>> jac(SM_DATA_D(J));

	// Actually get the Jacobian
	std::array<adept::adouble, Nspecies> y, dydt;

	adept::set_values(&y[0], Nspecies, NV_DATA_S(yv));

	sMem->stack.new_recording();

	// Get the data in the right form
	fullModel(y.data(), &rattes, dydt.data());

	sMem->stack.independent(&y[0], Nspecies);
	sMem->stack.dependent(&dydt[0], Nspecies);

	sMem->stack.jacobian(jac.data());

	return 0;
}


int JacB(double t, N_Vector y, N_Vector a, N_Vector b, SUNMatrix J, void *user_data, N_Vector c, N_Vector d, N_Vector e) {
	Jac(t, y, a, J, user_data, c, d, e);

	Eigen::Map<Eigen::Matrix<double, Nspecies, Nspecies>> jac(SM_DATA_D(J));

	jac = -jac;
	jac.transposeInPlace();

	return 0;
}


// fQB routine. Compute integrand for quadratures
static int fQB(double t, N_Vector y, N_Vector yB, N_Vector qBdot, void *user_dataB) {
	solver *sMem = static_cast<solver *>(user_dataB);
	const size_t Np = sMem->params.size();
	sMem->stack.activate();

	vector<adouble> X(Np);
	array<adouble, Nspecies> dydt;
	adept::set_values(&X[0], Np, sMem->params.data());

	if (t < sMem->preT)
		adept::set_values(&X[0], Nlig, sMem->preL);

	sMem->stack.new_recording();

	ratesS<adouble> rattes = ratesS<adouble>(X);

	// Get the data in the right form
	fullModel(NV_DATA_S(y), &rattes, dydt.data());

	adouble yOut = 0;
	yOut = std::inner_product(dydt.begin(), dydt.end(), NV_DATA_S(yB), yOut);

	sMem->stack.independent(&X[0], Np);
	sMem->stack.dependent(&yOut, 1);

	sMem->stack.jacobian(NV_DATA_S(qBdot));

	return(0);
}


static void errorHandler(int error_code, const char *module, const char *function, char *msg, void *ehdata) {
	if (error_code == CV_WARNING) return;
	solver *sMem = static_cast<solver *>(ehdata);
	ratesS<double> ratt = sMem->getRates();

	std::lock_guard<std::mutex> lock(print_mutex);

	cout << "Internal CVode error in " << function << ", module: " << module << ", error code: " << error_code << std::endl;
	cout << msg << std::endl;
	cout << "Parameters: ";

	for (size_t ii = 0; ii < Nparams; ii++) {
		cout << sMem->params[ii] << "\t";
	}
	
	ratt.print();

	if (sMem->sensi)
		cout << "Sensitivity enabled." << std::endl;

	cout << std::endl << std::endl;
}


int fullModelCVode(const double t, const N_Vector xx, N_Vector dxxdt, void *user_data) {
	solver *sMem = static_cast<solver *>(user_data);
	ratesS<double> rattes = sMem->getRates();

	if (t < sMem->preT)
		preTreat(t - sMem->preT, rattes.ILs, sMem->preL);

	// Get the data in the right form
	fullModel(NV_DATA_S(xx), &rattes, NV_DATA_S(dxxdt));

	return 0;
}


extern "C" int runCkine (const double * const tps, const size_t ntps, double * const out, const double * const rxnRatesIn, bool IL2case, const double preT, const double * const preL) {
	size_t itps = 0;

	std::vector<double> v;

	if (IL2case) {
		v = std::vector<double>(rxnRatesIn, rxnRatesIn + NIL2params);
	} else {
		v = std::vector<double>(rxnRatesIn, rxnRatesIn + Nparams);
	}

	solver sMem(v, preT, preL);

	if (tps[0] + preT < std::numeric_limits<double>::epsilon()) {
		std::copy_n(NV_DATA_S(sMem.state), Nspecies, out);

		itps = 1;
	}

	for (; itps < ntps; itps++) {
		if (sMem.CVodeRun(tps[itps] + preT) < 0) return -1;

		// Copy out result
		std::copy_n(NV_DATA_S(sMem.state), Nspecies, out + Nspecies*itps);
	}

	return 0;
}

x0JacM xNotp (vector<double> &params, adept::Stack *stack) {
	size_t Np = params.size();

	vector<adouble> X(Np);
	adept::set_values(&X[0], Np, params.data());

	stack->new_recording();

	ratesS<adouble> rattes = ratesS<adouble>(X);

	// Get the data in the right form
	std::array<adouble, Nspecies> outAD = solveAutocrine(&rattes);

	stack->independent(&X[0], Np);
	stack->dependent(&outAD[0], Nspecies);

	x0JacM gradZV(Nspecies, Np);
	stack->jacobian(gradZV.data());

	return gradZV;
}



extern "C" int runCkineS (const double * const tps, const size_t ntps, double * const out, double * const Sout, const double * const actV, const double * const rxnRatesIn, const bool IL2case, const double preT, const double * const preL) {
	size_t itps = 0;

	std::vector<double> v;
	std::array<double, Nspecies> actVv;
	std::copy_n(actV, Nspecies, actVv.begin());

	v = std::vector<double>(rxnRatesIn, rxnRatesIn + Nparams);

	solver sMem(v, actVv, preT, preL);

	if (tps[0] + preT < std::numeric_limits<double>::epsilon()) {
		out[0] = sMem.getActivity();

		itps = 1;
	}

	for (; itps < ntps; itps++) {
		if (sMem.CVodeRun(tps[itps] + preT) < 0) return -1;

		// Copy out result
		out[itps] = sMem.getActivity();
	}

	sMem.backward(tps[ntps-1] + preT);

	x0JacM x0p = xNotp(sMem.params, &sMem.stack);

	// Get sensitivities
	for (int bitps = ntps - 1; bitps >= 0; bitps--) {
		if (sMem.getAdjSens(tps[bitps] + preT, Sout + sMem.params.size()*bitps, x0p)) return -1;
	}

	return 0;
}


extern "C" int runCkineParallel (const double * const rxnRatesIn, const double * const tps, const size_t ntps, size_t nDoses, double *out, const double preT, const double * const preL) {
	ThreadPool pool;
	int retVal = 1000;
	std::list<std::future<int>> results;

	// Actually run the simulations
	for (size_t ii = 0; ii < nDoses; ii++)
		results.push_back(pool.enqueue(runCkine, tps, ntps, out + Nspecies*ii*ntps, rxnRatesIn + ii*Nparams, false, preT, preL));

	// Synchronize all threads
	for (std::future<int> &th:results) retVal = std::min(th.get(), retVal);

	// Get the worst case to return
	return retVal;
}


extern "C" int runCkineSParallel (const double * const rxnRatesIn, const double * const tps, const size_t ntps, const size_t nDoses, double * const out, double * const Sout, double * const actV, const double preT, const double * const preL) {
	ThreadPool pool;
	int retVal = 1000;
	std::list<std::future<int>> results;

	// Actually run the simulations
	for (size_t ii = 0; ii < nDoses; ii++) {
		results.push_back(pool.enqueue(runCkineS, tps, ntps, out + ii*ntps, Sout + Nparams*ii*ntps, actV, rxnRatesIn + Nparams*ii, false, preT, preL));
	}

	// Synchronize all threads
	for (std::future<int> &th:results) retVal = std::min(th.get(), retVal);

	// Get the worst case to return
	return retVal;
}
