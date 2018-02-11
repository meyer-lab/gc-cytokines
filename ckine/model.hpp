#include <array>

extern "C" int runCkine (double *, size_t, double *, double *, double *);


struct ratesS {
	std::array<double, 11> trafRates;
	std::array<double, 17> rxn;
};