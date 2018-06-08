#include <array>

struct ratesS {
	double IL2;
	double IL15;
	double IL7;
	double IL9;
	double kfwd;
	double k4rev;
	double k5rev;
	double k8rev;
	double k9rev;
	double k10rev;
	double k11rev;
	double k12rev;
	double k16rev;
	double k17rev;
	double k20rev;
	double k21rev;
	double k22rev;
	double k23rev;
	double k24rev;
	double k25rev;
	double k27rev;
	double k29rev;
	double k31rev;
	double endo;
	double activeEndo;
	double sortF;
	double kRec;
	double kDeg;
	std::array<double, 6> Rexpr;
};

// These are probably measured in the literature
constexpr double kfbnd = 0.60; // Assuming on rate of 10^7 M-1 sec-1
constexpr double k1rev = kfbnd * 10; // doi:10.1016/j.jmb.2004.04.038, 10 nM

constexpr double k2rev = kfbnd * 144; // doi:10.1016/j.jmb.2004.04.038, 144 nM

// Literature values for k values for IL-15
constexpr double k13rev = kfbnd * 0.065; // based on the multiple papers suggesting 30-100 pM
constexpr double k14rev = kfbnd * 438; // doi:10.1038/ni.2449, 438 nM

// Literature values for IL-7
constexpr double k25rev = kfbnd * 59; // DOI:10.1111/j.1600-065X.2012.01160.x, 59 nM

constexpr double abstolIn = 1E-2;
constexpr double reltolIn = 1E-3;
constexpr double internalV = 623.0; // Same as that used in TAM model
constexpr double internalFrac = 0.5; // Same as that used in TAM model

constexpr size_t Nparams = 23; // length of rxntfR vector
constexpr size_t Nspecies = 48; // number of complexes in surface + endosome + free ligand
constexpr size_t halfL = 22; // number of complexes on surface alone

extern "C" int runCkine (double *, size_t, double *, double *, bool, double *);
