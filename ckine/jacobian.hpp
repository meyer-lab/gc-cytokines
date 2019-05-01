#include <adept.h>
#include "reaction.hpp"


template <class T>
void fullJacobian(const double * const yv, ratesS<double> * const r, T &out, adept::Stack *stack) {
	std::array<adept::adouble, Nspecies> y, dydt;

	adept::set_values(&y[0], Nspecies, yv);

	stack->new_recording();

	// Get the data in the right form
	fullModel(y.data(), r, dydt.data());

	stack->independent(&y[0], Nspecies);
	stack->dependent(&dydt[0], Nspecies);

	stack->jacobian(out.data());
}
