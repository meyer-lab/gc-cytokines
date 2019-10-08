#include <array>
#include <string>

constexpr size_t Nparams = 30; // number of unknowns for the full model
constexpr size_t NIL2params = 15; // number of unknowns for the IL2 model including IL2's endosomal binding affinities

constexpr size_t Nlig = 6; // Number of ligands

// Measured in the literature
constexpr double kfbnd = 0.60; // Assuming on rate of 10^7 M-1 sec-1


template <class T>
struct bindingRates {
	T kfwd, k1rev, k2rev, k4rev, k5rev, k10rev;
	T k11rev, k13rev, k14rev, k16rev, k17rev;
	T k22rev, k23rev, k25rev, k27rev, k29rev;
	T k31rev, k32rev, k33rev, k34rev, k35rev;
};


template <class T>
class ratesS {
public:
	std::array<T, Nlig> ILs; // IL2, 15, 7, 9, 4, 21
	bindingRates<T> surface, endosome;
	T endo, activeEndo;
	T sortF, kRec, kDeg;
	std::array<T, 8> Rexpr; // Expression: IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R, IL4Ra, IL21Ra

	void setTraffic(T * traf) {
		// Set the rates
		endo = traf[0];
		activeEndo = traf[1];
		sortF = traf[2];
		kRec = traf[3];
		kDeg = traf[4];

		if (sortF > 1.0) {
			throw std::runtime_error(std::string("sortF is a fraction and cannot be greater than 1.0."));
		}
	}

	void endosomeAdjust(bindingRates<T> *endosome) {
		endosome->k1rev *= 5.0;
		endosome->k2rev *= 5.0;
		endosome->k4rev *= 5.0;
		endosome->k5rev *= 5.0;
		endosome->k10rev *= 5.0;
		endosome->k11rev *= 5.0;
		endosome->k13rev *= 5.0;
		endosome->k14rev *= 5.0;
		endosome->k16rev *= 5.0;
		endosome->k17rev *= 5.0;
		endosome->k22rev *= 5.0;
		endosome->k23rev *= 5.0;
		endosome->k25rev *= 5.0;
		endosome->k27rev *= 5.0;
		endosome->k29rev *= 5.0;
		endosome->k31rev *= 5.0;
		endosome->k32rev *= 5.0;
		endosome->k33rev *= 5.0;
		endosome->k34rev *= 5.0;
		endosome->k35rev *= 5.0;
	}

	explicit ratesS(std::vector<T> rxntfR) {
		if (rxntfR.size() == Nparams) {
			std::copy_n(rxntfR.begin(), ILs.size(), ILs.begin());
			surface.kfwd = rxntfR[6];
			surface.k1rev = kfbnd * 10; // doi:10.1016/j.jmb.2004.04.038, 10 nM
			surface.k2rev = kfbnd * 144; // doi:10.1016/j.jmb.2004.04.038, 144 nM
			surface.k4rev = rxntfR[7];
			surface.k5rev = rxntfR[8];
			surface.k10rev = 12.0 * surface.k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
			surface.k11rev = 63.0 * surface.k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
			surface.k13rev = kfbnd * 0.065; // based on the multiple papers suggesting 30-100 pM
			surface.k14rev = kfbnd * 438; // doi:10.1038/ni.2449, 438 nM
			surface.k16rev = rxntfR[9];
			surface.k17rev = rxntfR[10];
			surface.k22rev = rxntfR[11];
			surface.k23rev = rxntfR[12];
			surface.k25rev = kfbnd * 59; // DOI:10.1111/j.1600-065X.2012.01160.x, 59 nM
			surface.k27rev = rxntfR[13];
			surface.k29rev = kfbnd * 0.1; // DOI:10.1073/pnas.89.12.5690, ~100 pM
			surface.k31rev = rxntfR[14];
			surface.k32rev = kfbnd * 1.0; // DOI: 10.1126/scisignal.aal1253 (human)
			surface.k33rev = rxntfR[15];
			surface.k34rev = kfbnd * 0.07; // DOI: 10.1126/scisignal.aal1253 (human)
			surface.k35rev = rxntfR[16];

			setTraffic(rxntfR.data() + 17);

			std::copy_n(rxntfR.data() + 22, 8, Rexpr.begin());

			// all reverse rates are 5x larger in the endosome
			endosome = surface;
			endosomeAdjust(&endosome);
		} else {
			std::fill(ILs.begin(), ILs.end(), 0.0);
			ILs[0] = rxntfR[0];
			surface.kfwd = rxntfR[1];
			surface.k1rev = rxntfR[2];
			surface.k2rev = rxntfR[3];
			surface.k4rev = rxntfR[4];
			surface.k5rev = rxntfR[5];
			surface.k11rev = rxntfR[6];
			surface.k13rev = 1.0;
			surface.k14rev = 1.0;
			surface.k16rev = 1.0;
			surface.k17rev = 1.0;
			surface.k22rev = 1.0;
			surface.k23rev = 1.0;
			surface.k25rev = 1.0;
			surface.k27rev = 1.0;
			surface.k29rev = 1.0;
			surface.k31rev = 1.0;
			surface.k32rev = 1.0;
			surface.k33rev = 1.0;
			surface.k34rev = 1.0;
			surface.k35rev = 1.0;

			// These are probably measured in the literature
			surface.k10rev = 12.0 * surface.k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038

			std::array<T, 5> trafP = {0.08221, 2.52654, 0.16024, 0.10017, 0.00807};

			setTraffic(trafP.data());

			std::fill(Rexpr.begin(), Rexpr.end(), 0.0);
			Rexpr[0] = rxntfR[7];
			Rexpr[1] = rxntfR[8];
			Rexpr[2] = rxntfR[9];

			endosome = surface;

			// manually reassign all of IL2's endosomal binding affinities
			endosome.k1rev = rxntfR[10];
			endosome.k2rev = rxntfR[11];
			endosome.k4rev = rxntfR[12];
			endosome.k5rev = rxntfR[13];
			endosome.k11rev = rxntfR[14];
			endosome.k10rev = 12.0 * endosome.k5rev / 1.5; // doi:10.1016/j.jmb.2004.04.038
		}
	}

	void print() {
		std::cout << "kfwd: " << surface.kfwd << std::endl;
		std::cout << "k4rev: " << surface.k4rev << std::endl;
		std::cout << "k5rev: " << surface.k5rev << std::endl;
		std::cout << "k10rev: " << surface.k10rev << std::endl;
		std::cout << "k11rev: " << surface.k11rev << std::endl;
		std::cout << "k16rev: " << surface.k16rev << std::endl;
		std::cout << "k17rev: " << surface.k17rev << std::endl;
		std::cout << "k22rev: " << surface.k22rev << std::endl;
		std::cout << "k23rev: " << surface.k23rev << std::endl;
		std::cout << "k27rev: " << surface.k27rev << std::endl;
		std::cout << "k31rev: " << surface.k31rev << std::endl;
		std::cout << "k33rev: " << surface.k33rev << std::endl;
		std::cout << "k35rev: " << surface.k35rev << std::endl;
		std::cout << "endo: " << endo << std::endl;
		std::cout << "activeEndo: " << activeEndo << std::endl;
		std::cout << "sortF: " << sortF << std::endl;
		std::cout << "kRec: " << kRec << std::endl;
		std::cout << "kDeg: " << kDeg << std::endl;
	}
};


constexpr double internalV = 623.0; // Same as that used in TAM model
constexpr double internalFrac = 0.5; // Same as that used in TAM model

constexpr size_t Nspecies = 62; // number of complexes in surface + endosome + free ligand
constexpr size_t halfL = 28; // number of complexes on surface alone

extern "C" int runCkine (const double *tps, size_t ntps, double *out, const double * const rxnRatesIn, bool, const double preT, const double * const preL);
extern "C" int runCkineS (const double * const tps, const size_t ntps, double * const out, double * const Sout, const double * const actV, const double * const rxnRatesIn, const bool IL2case, const double preT, const double * const preL);
