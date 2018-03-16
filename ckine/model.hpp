#include <array>

struct ratesS {
	double IL2;
	double IL15;
	double IL7;
	double IL9;
	double kfwd;
	double k5rev;
	double k6rev;
	double k15rev;
	double k17rev;
	double k18rev;
	double k22rev;
	double k23rev;
	double k26rev;
	double k27rev;
	double k29rev;
	double k30rev;
	double k31rev;
	double kfbnd;
	double k1rev;
	double k2rev;
	double k3fwd;
	double k3rev;
	double k10rev;
	double k11rev;
	double k13rev;
	double k14rev;
	double k25rev;
	double k4rev;
	double k7rev;
	double k12rev;
	double k9rev;
	double k8rev;
	double k16rev;
	double k19rev;
	double k24rev;
	double k21rev;
	double k20rev;
	double k32rev;
	double k28rev;
	double endo;
	double activeEndo;
	double sortF;
	double kRec;
	double kDeg;
	std::array<double, 6> Rexpr;
};

const double abstolIn = 1E-5;
const double reltolIn = 1E-7;
const double internalV = 623.0; // Same as that used in TAM model
const double internalFrac = 0.5; // Same as that used in TAM model

// The indices carried over in the reduced IL2 model
const std::array<size_t, 21> IL2_assoc = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 52}};

std::array<bool, 26> __active_species_IDX() {
	std::array<bool, 26> __active_species_IDX;
	std::fill(__active_species_IDX.begin(), __active_species_IDX.end(), false);

	__active_species_IDX[8] = true;
	__active_species_IDX[9] = true;
	__active_species_IDX[16] = true;
	__active_species_IDX[17] = true;
	__active_species_IDX[21] = true;
	__active_species_IDX[25] = true;

	return __active_species_IDX;
}

const std::array<bool, 26> activeV = __active_species_IDX();