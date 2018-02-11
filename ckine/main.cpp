#include <iostream>
#include "model.hpp"
#include <random>
#include <array>

using namespace std;

int main() {

	ratesS rrates;

	default_random_engine gen;
	uniform_real_distribution<double> uniR(0.0, 1.0);
	lognormal_distribution<double> logD(0.0, 1.0);

	array<double, 4> tps = {1.0, 10.0, 100.0, 1000.0};

	array<double, 4*56> out;

	auto logDl = [&, logD]() mutable { return logD(gen); };


	for (size_t ii = 0; ii < 5000; ii++) {
		generate(rrates.trafRates.begin(), rrates.trafRates.end(), logDl);
		generate(rrates.rxn.begin(), rrates.rxn.end(), logDl);
		rrates.trafRates[2] = uniR(gen);

		rrates.trafRates[8] = 0.0;
		rrates.trafRates[9] = 0.0;
		rrates.trafRates[10] = 0.0;

		int retVal = runCkine (tps.data(), tps.size(), out.data(), rrates.rxn.data(), rrates.trafRates.data());

		if (retVal < 0) {
			cout << ii << " failed. Code: " << retVal << endl;
		}
	}

	return 0;
}