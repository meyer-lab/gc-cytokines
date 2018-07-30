#include <array>
#include <string>


// Measured in the literature
constexpr double kfbnd = 0.60; // Assuming on rate of 10^7 M-1 sec-1
constexpr double k1rev = kfbnd * 10; // doi:10.1016/j.jmb.2004.04.038, 10 nM

constexpr double k2rev = kfbnd * 144; // doi:10.1016/j.jmb.2004.04.038, 144 nM

// Literature values for k values for IL-15
constexpr double k13rev = kfbnd * 0.065; // based on the multiple papers suggesting 30-100 pM
constexpr double k14rev = kfbnd * 438; // doi:10.1038/ni.2449, 438 nM

// Literature values for IL-7
constexpr double k25rev = kfbnd * 59; // DOI:10.1111/j.1600-065X.2012.01160.x, 59 nM

// Literature value for IL-9
constexpr double k29rev = kfbnd * 0.1; // DOI:10.1073/pnas.89.12.5690, ~100 pM

// Literature value for IL-4
constexpr double k32rev = kfbnd * 1.0; // DOI: 10.1126/scisignal.aal1253 (human)

// Literature value for IL-21
constexpr double k34rev = kfbnd * 0.07; // DOI: 10.1126/scisignal.aal1253 (human)


class ratesS {
public:
	std::array<double, 6> ILs; // IL2, 15, 7, 9, 4, 21
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
	double k27rev;
	double k31rev;
	double k33rev;
	double k35rev;
	double endo;
	double activeEndo;
	double sortF;
	double kRec;
	double kDeg;
	std::array<double, 8> Rexpr;

	explicit ratesS(const double * const rxntfR) {
		std::copy_n(rxntfR, ILs.size(), ILs.begin());
		kfwd = rxntfR[6];
		k4rev = rxntfR[7];
		k5rev = rxntfR[8];
		k16rev = rxntfR[9];
		k17rev = rxntfR[10];
		k22rev = rxntfR[11];
		k23rev = rxntfR[12];
		k27rev = rxntfR[13];
		k31rev = rxntfR[14];
		k33rev = rxntfR[15];
		k35rev = rxntfR[16];

		// These are probably measured in the literature
		k10rev = 12.0 * k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
		k11rev = 63.0 * k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
		// To satisfy detailed balance these relationships should hold
		// Based on initial assembly steps
		k12rev = k1rev * k11rev / k2rev; // loop for IL2_IL2Ra_IL2Rb
		// Based on formation of full complex (IL2_IL2Ra_IL2Rb_gc)
		k9rev = k10rev * k11rev / k4rev;
		k8rev = k10rev * k12rev / k5rev;

		// IL15
		// To satisfy detailed balance these relationships should hold
		// _Based on initial assembly steps
		k24rev = k13rev * k23rev / k14rev; // loop for IL15_IL15Ra_IL2Rb still holds

		// _Based on formation of full complex
		k21rev = k22rev * k23rev / k16rev;
		k20rev = k22rev * k24rev / k17rev;

		// Set the rates
		endo = rxntfR[17];
		activeEndo = rxntfR[18];
		sortF = rxntfR[19];
		kRec = rxntfR[20];
		kDeg = rxntfR[21];

		if (sortF > 1.0) {
			throw std::runtime_error(std::string("sortF is a fraction and cannot be greater than 1.0."));
		}

		// Expression: IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R, IL4Ra, IL21Ra
		std::copy_n(rxntfR + 22, 8, Rexpr.begin());
	}

	void print() {
		std::cout << "kfwd: " << kfwd << std::endl;
		std::cout << "k4rev: " << k4rev << std::endl;
		std::cout << "k5rev: " << k5rev << std::endl;
		std::cout << "k8rev: " << k8rev << std::endl;
		std::cout << "k9rev: " << k9rev << std::endl;
		std::cout << "k10rev: " << k10rev << std::endl;
		std::cout << "k11rev: " << k11rev << std::endl;
		std::cout << "k12rev: " << k12rev << std::endl;
		std::cout << "k16rev: " << k16rev << std::endl;
		std::cout << "k17rev: " << k17rev << std::endl;
		std::cout << "k20rev: " << k20rev << std::endl;
		std::cout << "k21rev: " << k21rev << std::endl;
		std::cout << "k22rev: " << k22rev << std::endl;
		std::cout << "k23rev: " << k23rev << std::endl;
		std::cout << "k24rev: " << k24rev << std::endl;
		std::cout << "k27rev: " << k27rev << std::endl;
		std::cout << "k31rev: " << k31rev << std::endl;
		std::cout << "k33rev: " << k33rev << std::endl;
		std::cout << "k35rev: " << k35rev << std::endl;
		std::cout << "endo: " << endo << std::endl;
		std::cout << "activeEndo: " << activeEndo << std::endl;
		std::cout << "sortF: " << sortF << std::endl;
		std::cout << "kRec: " << kRec << std::endl;
		std::cout << "kDeg: " << kDeg << std::endl;
	}
};


constexpr double tolIn = 1.5E-6;
constexpr double internalV = 623.0; // Same as that used in TAM model
constexpr double internalFrac = 0.5; // Same as that used in TAM model

constexpr size_t Nparams = 30; // length of rxntfR vector
constexpr size_t Nspecies = 62; // number of complexes in surface + endosome + free ligand
constexpr size_t halfL = 28; // number of complexes on surface alone

extern "C" int runCkine (double *tps, size_t ntps, double *out, const double * const rxnRatesIn, const bool sensi, double *sensiOut);
