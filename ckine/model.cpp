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

using std::array;
using std::copy;
using std::copy_n;
using std::vector;
using std::fill;
using std::string;
using std::runtime_error;

struct ratesS {
	std::array<double, 11> trafRates;
	std::array<double, 17> rxn;
};

const double abstolIn = 1E-5;
const double reltolIn = 1E-7;
const double internalV = 623.0; // Same as that used in TAM model
const double internalFrac = 0.5; // Same as that used in TAM model

// The indices carried over in the reduced IL2 model
const array<size_t, 21> IL2_assoc = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 52}};

array<bool, 26> __active_species_IDX() {
	array<bool, 26> __active_species_IDX;
	fill(__active_species_IDX.begin(), __active_species_IDX.end(), false);

	__active_species_IDX[8] = true;
	__active_species_IDX[9] = true;
	__active_species_IDX[16] = true;
	__active_species_IDX[17] = true;
	__active_species_IDX[21] = true;
	__active_species_IDX[25] = true;

	return __active_species_IDX;
}

const array<bool, 26> activeV = __active_species_IDX();


void dy_dt(const double * const y, const double * const rxn, double *dydt) {
	// Set the constant inputs
	double IL2 = rxn[0];
	double IL15 = rxn[1];
	double IL7 = rxn[2];
	double IL9 = rxn[3];
	double kfwd = rxn[4];
	double k5rev = rxn[5];
	double k6rev = rxn[6];
	double k15rev = rxn[7];
	double k17rev = rxn[8];
	double k18rev = rxn[9];
	double k22rev = rxn[10];
	double k23rev = rxn[11];
	double k26rev = rxn[12];
	double k27rev = rxn[13];
	double k29rev = rxn[14];
	double k30rev = rxn[15];
	double k31rev = rxn[16];

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

	// These are probably measured in the literature
	double kfbnd = 0.01; // Assuming on rate of 10^7 M-1 sec-1
	double k1rev = kfbnd * 10; // doi:10.1016/j.jmb.2004.04.038, 10 nM

	double k2rev = kfbnd * 144; // doi:10.1016/j.jmb.2004.04.038, 144 nM
	double k3fwd = kfbnd / 10.0; // Very weak, > 50 uM. Voss, et al (1993). PNAS. 90, 2428–2432.
	double k3rev = 50000 * k3fwd;
	double k10rev = 12.0 * k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
	double k11rev = 63.0 * k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
	
	// Literature values for k values for IL-15
	double k13rev = kfbnd * 0.065; // based on the multiple papers suggesting 30-100 pM
	double k14rev = kfbnd * 438; // doi:10.1038/ni.2449, 438 nM
	
	// Literature values for IL-7
	double k25rev = kfbnd * 59; // DOI:10.1111/j.1600-065X.2012.01160.x, 59 nM
	
	// To satisfy detailed balance these relationships should hold
	// _Based on initial assembly steps
	double k4rev = kfbnd * k6rev * k3rev / k1rev / k3fwd;
	double k7rev = k3fwd * k2rev * k5rev / kfbnd / k3rev;
	double k12rev = k1rev * k11rev / k2rev;
	// _Based on formation of full complex
	double k9rev = k2rev * k10rev * k12rev / kfbnd / k3rev / k6rev * k3fwd;
	double k8rev = k2rev * k10rev * k12rev / kfbnd / k7rev / k3rev * k3fwd;

	// IL15
	// To satisfy detailed balance these relationships should hold
	// _Based on initial assembly steps
	double k16rev = kfwd * k18rev * k15rev / k13rev / kfbnd;
	double k19rev = kfwd * k14rev * k17rev / kfbnd / k15rev;
	double k24rev = k13rev * k23rev / k14rev;
	// _Based on formation of full complex

	double k21rev = k14rev * k22rev * k24rev / kfwd / k15rev / k18rev * kfbnd;
	double k20rev = k14rev * k22rev * k24rev / k19rev / k15rev;

	// _One detailed balance IL7/9 loop
	double k32rev = k29rev * k31rev / k30rev;
	double k28rev = k25rev * k27rev / k26rev;
	
	// IL2
	dydt[0] = -kfbnd * IL2Ra * IL2 + k1rev * IL2_IL2Ra - kfwd * IL2Ra * IL2_gc + k6rev * IL2_IL2Ra_gc - kfwd * IL2Ra * IL2_IL2Rb_gc + k8rev * IL2_IL2Ra_IL2Rb_gc - kfwd * IL2Ra * IL2_IL2Rb + k12rev * IL2_IL2Ra_IL2Rb;
	dydt[1] = -kfbnd * IL2Rb * IL2 + k2rev * IL2_IL2Rb - kfwd * IL2Rb * IL2_gc + k7rev * IL2_IL2Rb_gc - kfwd * IL2Rb * IL2_IL2Ra_gc + k9rev * IL2_IL2Ra_IL2Rb_gc - kfwd * IL2Rb * IL2_IL2Ra + k11rev * IL2_IL2Ra_IL2Rb;
	dydt[2] = -k3fwd * IL2 * gc + k3rev * IL2_gc - kfwd * IL2_IL2Rb * gc + k5rev * IL2_IL2Rb_gc - kfwd * IL2_IL2Ra * gc + k4rev * IL2_IL2Ra_gc - kfwd * IL2_IL2Ra_IL2Rb * gc + k10rev * IL2_IL2Ra_IL2Rb_gc;
	dydt[3] = -kfwd * IL2_IL2Ra * IL2Rb + k11rev * IL2_IL2Ra_IL2Rb - kfwd * IL2_IL2Ra * gc + k4rev * IL2_IL2Ra_gc + kfbnd * IL2 * IL2Ra - k1rev * IL2_IL2Ra;
	dydt[4] = -kfwd * IL2_IL2Rb * IL2Ra + k12rev * IL2_IL2Ra_IL2Rb - kfwd * IL2_IL2Rb * gc + k5rev * IL2_IL2Rb_gc + kfbnd * IL2 * IL2Rb - k2rev * IL2_IL2Rb;
	dydt[5] = -kfwd *IL2_gc * IL2Ra + k6rev * IL2_IL2Ra_gc - kfwd * IL2_gc * IL2Rb + k7rev * IL2_IL2Rb_gc + k3fwd * IL2 * gc - k3rev * IL2_gc;
	dydt[6] = -kfwd * IL2_IL2Ra_IL2Rb * gc + k10rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra * IL2Rb - k11rev * IL2_IL2Ra_IL2Rb + kfwd * IL2_IL2Rb * IL2Ra - k12rev * IL2_IL2Ra_IL2Rb;
	dydt[7] = -kfwd * IL2_IL2Ra_gc * IL2Rb + k9rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra * gc - k4rev * IL2_IL2Ra_gc + kfwd * IL2_gc * IL2Ra - k6rev * IL2_IL2Ra_gc;
	dydt[8] = -kfwd * IL2_IL2Rb_gc * IL2Ra + k8rev * IL2_IL2Ra_IL2Rb_gc + kfwd * gc * IL2_IL2Rb - k5rev * IL2_IL2Rb_gc + kfwd * IL2_gc * IL2Rb - k7rev * IL2_IL2Rb_gc;
	dydt[9] = kfwd * IL2_IL2Rb_gc * IL2Ra - k8rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra_gc * IL2Rb - k9rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra_IL2Rb * gc - k10rev * IL2_IL2Ra_IL2Rb_gc;

	// IL15
	dydt[10] = -kfbnd * IL15Ra * IL15 + k13rev * IL15_IL15Ra - kfbnd * IL15Ra * IL15_gc + k18rev * IL15_IL15Ra_gc - kfwd * IL15Ra * IL15_IL2Rb_gc + k20rev * IL15_IL15Ra_IL2Rb_gc - kfwd * IL15Ra * IL15_IL2Rb + k24rev * IL15_IL15Ra_IL2Rb;
	dydt[11] = -kfwd * IL15_IL15Ra * IL2Rb + k23rev * IL15_IL15Ra_IL2Rb - kfwd * IL15_IL15Ra * gc + k16rev * IL15_IL15Ra_gc + kfbnd * IL15 * IL15Ra - k13rev * IL15_IL15Ra;
	dydt[12] = -kfwd * IL15_IL2Rb * IL15Ra + k24rev * IL15_IL15Ra_IL2Rb - kfbnd * IL15_IL2Rb * gc + k17rev * IL15_IL2Rb_gc + kfbnd * IL15 * IL2Rb - k14rev * IL15_IL2Rb;
	dydt[13] = -kfbnd * IL15_gc * IL15Ra + k18rev * IL15_IL15Ra_gc - kfwd * IL15_gc * IL2Rb + k19rev * IL15_IL2Rb_gc + kfbnd * IL15 * gc - k15rev * IL15_gc;
	dydt[14] = -kfwd * IL15_IL15Ra_IL2Rb * gc + k22rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra * IL2Rb - k23rev * IL15_IL15Ra_IL2Rb + kfwd * IL15_IL2Rb * IL15Ra - k24rev * IL15_IL15Ra_IL2Rb;
	dydt[15] = -kfwd * IL15_IL15Ra_gc * IL2Rb + k21rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra * gc - k16rev * IL15_IL15Ra_gc + kfbnd * IL15_gc * IL15Ra - k18rev * IL15_IL15Ra_gc;
	dydt[16] = -kfwd * IL15_IL2Rb_gc * IL15Ra + k20rev * IL15_IL15Ra_IL2Rb_gc + kfbnd * gc * IL15_IL2Rb - k17rev * IL15_IL2Rb_gc + kfwd * IL15_gc * IL2Rb - k19rev * IL15_IL2Rb_gc;
	dydt[17] =  kfwd * IL15_IL2Rb_gc * IL15Ra - k20rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra_gc * IL2Rb - k21rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra_IL2Rb * gc - k22rev * IL15_IL15Ra_IL2Rb_gc;
	
	dydt[1] = dydt[1] - kfbnd * IL2Rb * IL15 + k14rev * IL15_IL2Rb - kfwd * IL2Rb * IL15_gc + k19rev * IL15_IL2Rb_gc - kfwd * IL2Rb * IL15_IL15Ra_gc + k21rev * IL15_IL15Ra_IL2Rb_gc - kfwd * IL2Rb * IL15_IL15Ra + k23rev * IL15_IL15Ra_IL2Rb;
	dydt[2] = dydt[2] - kfbnd * IL15 * gc + k15rev * IL15_gc - kfbnd * IL15_IL2Rb * gc + k17rev * IL15_IL2Rb_gc - kfwd * IL15_IL15Ra * gc + k16rev * IL15_IL15Ra_gc - kfwd * IL15_IL15Ra_IL2Rb * gc + k22rev * IL15_IL15Ra_IL2Rb_gc;
	
	// IL7
	dydt[2] = dydt[2] - kfbnd * IL7 * gc + k26rev * gc_IL7 - kfwd * gc * IL7Ra_IL7 + k27rev * IL7Ra_gc_IL7;
	dydt[18] = -kfbnd * IL7Ra * IL7 + k25rev * IL7Ra_IL7 - kfwd * IL7Ra * gc_IL7 + k28rev * IL7Ra_gc_IL7;
	dydt[19] = kfbnd * IL7Ra * IL7 - k25rev * IL7Ra_IL7 - kfwd * gc * IL7Ra_IL7 + k27rev * IL7Ra_gc_IL7;
	dydt[20] = -kfwd * IL7Ra * gc_IL7 + k28rev * IL7Ra_gc_IL7 + kfbnd * IL7 * gc - k26rev * gc_IL7;
	dydt[21] = kfwd * IL7Ra * gc_IL7 - k28rev * IL7Ra_gc_IL7 + kfwd * gc * IL7Ra_IL7 - k27rev * IL7Ra_gc_IL7;

	// IL9
	dydt[2] = dydt[2] - kfbnd * IL9 * gc + k30rev * gc_IL9 - kfwd * gc * IL9R_IL9 + k31rev * IL9R_gc_IL9;
	dydt[22] = -kfbnd * IL9R * IL9 + k29rev * IL9R_IL9 - kfwd * IL9R * gc_IL9 + k32rev * IL9R_gc_IL9;
	dydt[23] = kfbnd * IL9R * IL9 - k29rev * IL9R_IL9 - kfwd * gc * IL9R_IL9 + k31rev * IL9R_gc_IL9;
	dydt[24] = -kfwd * IL9R * gc_IL9 + k32rev * IL9R_gc_IL9 + kfbnd * IL9 * gc - k30rev * gc_IL9;
	dydt[25] = kfwd * IL9R * gc_IL9 - k32rev * IL9R_gc_IL9 + kfwd * gc * IL9R_IL9 - k31rev * IL9R_gc_IL9;
}


extern "C" void dydt_C(double *y_in, double t, double *dydt_out, double *rxn_in) {
	dy_dt(y_in, rxn_in, dydt_out);
}


void findLigConsume(double *dydt) {
	// Calculate the ligand consumption.
	dydt[52] -= std::accumulate(dydt+3, dydt+10, 0) / internalV;
	dydt[53] -= std::accumulate(dydt+11, dydt+18, 0) / internalV;
	dydt[54] -= std::accumulate(dydt+19, dydt+22, 0) / internalV;
	dydt[55] -= std::accumulate(dydt+23, dydt+26, 0) / internalV;
}


void trafficking(const double * const y, array<double, 11> tfR, double *dydt) {
	// Implement trafficking.

	// Set the rates
	double endo = tfR[0];
	double activeEndo = tfR[1];
	double sortF = tfR[2];
	double kRec = tfR[3];
	double kDeg = tfR[4];

	size_t halfL = activeV.size();

	// Actually calculate the trafficking
	for (size_t ii = 0; ii < halfL; ii++) {
		if (activeV[ii]) {
			dydt[ii] += -y[ii]*(endo + activeEndo); // Endocytosis
			dydt[ii+halfL] += y[ii]*(endo + activeEndo)/internalFrac - kDeg*y[ii+halfL]; // Endocytosis, degradation
		} else {
			dydt[ii] += -y[ii]*endo + kRec*(1.0-sortF)*y[ii+halfL]*internalFrac; // Endocytosis, recycling
			dydt[ii+halfL] += y[ii]*endo/internalFrac - kRec*(1.0-sortF)*y[ii+halfL] - (kDeg*sortF)*y[ii+halfL]; // Endocytosis, recycling, degradation
		}
	}

	// Expression: IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R
	dydt[0] += tfR[5];
	dydt[1] += tfR[6];
	dydt[2] += tfR[7];
	dydt[10] += tfR[8];
	dydt[18] += tfR[9];
	dydt[22] += tfR[10];

	// Degradation does lead to some clearance of ligand in the endosome
	for (size_t ii = 0; ii < 4; ii++) {
		dydt[52 + ii] -= dydt[52 + ii] * kDeg;
	}
}


void fullModel(const double * const y, const array<double, 17> r, array<double, 11> tfR, double *dydt) {
	// Implement full model.
	fill(dydt, dydt + 56, 0.0);

	// Calculate endosomal reactions
	array<double, 17> rr = r;
	copy_n(y + 52, 4, rr.begin());

	// Calculate cell surface and endosomal reactions
	dy_dt(y, r.data(), dydt);
	dy_dt(y + 26, rr.data(), dydt + 26);

	// Handle trafficking
	trafficking(y, tfR, dydt);

	// Handle endosomal ligand balance.
	findLigConsume(dydt);
}


int fullModelCVode (const double, const N_Vector xx, N_Vector dxxdt, void *user_data) {
	ratesS *rIn = static_cast<ratesS *>(user_data);

	array<double, 56> xxArr;

	// Get the data in the right form
	if (NV_LENGTH_S(xx) == xxArr.size()) { // If we're using the full model
		fullModel(xxArr.data(), rIn->rxn, rIn->trafRates, NV_DATA_S(dxxdt));
	} else if (NV_LENGTH_S(xx) == IL2_assoc.size()) { // If it looks like we're using the IL2 model
		fill(xxArr.begin(), xxArr.end(), 0.0);

		for (size_t ii = 0; ii < IL2_assoc.size(); ii++) {
			xxArr[IL2_assoc[ii]] = NV_Ith_S(xx, ii);
		}

		array<double, 56> dydt;

		fullModel(xxArr.data(), rIn->rxn, rIn->trafRates, dydt.data());

		for (size_t ii = 0; ii < IL2_assoc.size(); ii++) {
			NV_Ith_S(dxxdt, ii) = dydt[IL2_assoc[ii]];
		}
	} else {
		throw runtime_error(string("Failed to find the right wrapper."));
	}

	return 0;
}


extern "C" void fullModel_C(const double * const y_in, double t, double *dydt_out, double *rxn_in, double *tfr_in) {
	// Bring back the wrapper!

	array<double, 17> r;
	array<double, 11> tf;

	copy_n(rxn_in, r.size(), r.begin());
	copy_n(tfr_in, tf.size(), tf.begin());

	fullModel(y_in, r, tf, dydt_out);
}


array<double, 56> solveAutocrine(array<double, 11> trafRates) {
	array<double, 56> y0;
	fill(y0.begin(), y0.end(), 0.0);

	array<size_t, 6> recIDX = {{0, 1, 2, 10, 18, 22}};

	double internalFrac = 0.5; // Same as that used in TAM model

	// Expand out trafficking terms
	double endo = trafRates[0];
	double sortF = trafRates[2];
	double kRec = trafRates[3]*(1-sortF);
	double kDeg = trafRates[4]*sortF;

	// Assuming no autocrine ligand, so can solve steady state
	// Add the species
	for (size_t ii = 0; ii < recIDX.size(); ii++) {
		y0[recIDX[ii] + 26] = trafRates[5 + ii] / kDeg / internalFrac;
		y0[recIDX[ii]] = (trafRates[5 + ii] + kRec*y0[recIDX[ii] + 26]*internalFrac)/endo;
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
		throw runtime_error(string("Error calling CVodeCreate in solver_setup."));
	}
	
	CVodeSetErrHandlerFn(cvode_mem, &errorHandler, params);

	/* Call CVodeInit to initialize the integrator memory and specify the
	 * user's right hand side function in y'=f(t,y), the inital time T0, and
	 * the initial dependent variable vector y. */
	if (CVodeInit(cvode_mem, fullModelCVode, 0.0, init) < 0) {
		CVodeFree(&cvode_mem);
		throw runtime_error(string("Error calling CVodeInit in solver_setup."));
	}
	
	N_Vector abbstol = N_VNew_Serial(NV_LENGTH_S(init));
	N_VConst(abstolIn, abbstol);
	
	/* Call CVodeSVtolerances to specify the scalar relative tolerance
	 * and vector absolute tolerances */
	if (CVodeSVtolerances(cvode_mem, reltolIn, abbstol) < 0) {
		N_VDestroy_Serial(abbstol);
		CVodeFree(&cvode_mem);
		throw runtime_error(string("Error calling CVodeSVtolerances in solver_setup."));
	}
	N_VDestroy_Serial(abbstol);

	SUNMatrix A = SUNDenseMatrix((int) NV_LENGTH_S(init), (int) NV_LENGTH_S(init));

	SUNLinearSolver LS = SUNDenseLinearSolver(init, A);
	
	// Call CVDense to specify the CVDENSE dense linear solver
	if (CVDlsSetLinearSolver(cvode_mem, LS, A) < 0) {
		CVodeFree(&cvode_mem);
		throw runtime_error(string("Error calling CVDense in solver_setup."));
	}
	
	// Pass along the parameter structure to the differential equations
	if (CVodeSetUserData(cvode_mem, params) < 0) {
		CVodeFree(&cvode_mem);
		throw runtime_error(string("Error calling CVodeSetUserData in solver_setup."));
	}

	CVodeSetMaxNumSteps(cvode_mem, 2000000);
	
	return cvode_mem;
}


extern "C" int runCkine (double *tps, size_t ntps, double *out, double *rxnRatesIn, double *trafRatesIn) {
	ratesS rattes;

	copy_n(rxnRatesIn, rattes.rxn.size(), rattes.rxn.begin());
	copy_n(trafRatesIn, rattes.trafRates.size(), rattes.trafRates.begin());

	array<double, 56> y0 = solveAutocrine(rattes.trafRates);
	N_Vector state;

	// Fill output values with 0's
	fill(out, out + ntps*y0.size(), 0.0);

	// Can we use the reduced IL2 only model
	if (rattes.trafRates[8] + rattes.trafRates[9] + rattes.trafRates[10] == 0.0) {
		state = N_VNew_Serial((long) IL2_assoc.size());

		for (size_t ii = 0; ii < IL2_assoc.size(); ii++) {
			NV_Ith_S(state, ii) = y0[IL2_assoc[ii]];
		}
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
