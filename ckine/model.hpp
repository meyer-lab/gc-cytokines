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
constexpr double kfbnd = 0.01; // Assuming on rate of 10^7 M-1 sec-1
constexpr double k1rev = kfbnd * 10; // doi:10.1016/j.jmb.2004.04.038, 10 nM

constexpr double k2rev = kfbnd * 144; // doi:10.1016/j.jmb.2004.04.038, 144 nM
constexpr double k3fwd = kfbnd / 10.0; // Very weak, > 50 uM. Voss, et al (1993). PNAS. 90, 2428–2432.
constexpr double k3rev = 50000 * k3fwd;

// Literature values for k values for IL-15
constexpr double k13rev = kfbnd * 0.065; // based on the multiple papers suggesting 30-100 pM
constexpr double k14rev = kfbnd * 438; // doi:10.1038/ni.2449, 438 nM

// Literature values for IL-7
constexpr double k25rev = kfbnd * 59; // DOI:10.1111/j.1600-065X.2012.01160.x, 59 nM
constexpr double k26rev = 50000 * kfbnd; // General assumption that cytokine doesn't bind to free gc

// Literature values for IL-9
constexpr double k30rev = 50000 * kfbnd; // General assumption that cytokine doesn't bind to free gc

constexpr double abstolIn = 1E-4;
constexpr double reltolIn = 1E-7;
constexpr double internalV = 623.0; // Same as that used in TAM model
constexpr double internalFrac = 0.5; // Same as that used in TAM model

constexpr size_t Nparams = 26;
constexpr size_t Nspecies = 56;

extern "C" int runCkine (double *, size_t, double *, double *, bool, double *);
