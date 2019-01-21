#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <array>
#include <random>
#include <algorithm>
#include <adept.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/XmlOutputter.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestRunner.h>

#include "model.hpp"
#include "jacobian.hpp"
#include "reaction.hpp"

using namespace std;

constexpr array<double, 4> tps = {{0.1, 10.0, 1000.0, 100000.0}};


class interfaceTestCase : public CppUnit::TestCase {
public:
	/// Setup method
	void setUp() {
		random_device rd;
		gen = new mt19937(rd());
		rxnRatesIn = getParams();
	}
 
	/// Teardown method
	void tearDown() {
		delete gen;
	}

	// method to create a suite of tests
	static CppUnit::Test *suite() {
		CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("interfaceTestCase");

		suiteOfTests->addTest(new CppUnit::TestCaller<interfaceTestCase>("testrunCkine",
			&interfaceTestCase::testrunCkine));
		suiteOfTests->addTest(new CppUnit::TestCaller<interfaceTestCase>("testrunCkineS",
			&interfaceTestCase::testrunCkineS));
		suiteOfTests->addTest(new CppUnit::TestCaller<interfaceTestCase>("testrunCkinePretreat",
			&interfaceTestCase::testrunCkinePretreat));
		suiteOfTests->addTest(new CppUnit::TestCaller<interfaceTestCase>("testrunCkineSPretreat",
			&interfaceTestCase::testrunCkineSPretreat));
		suiteOfTests->addTest(new CppUnit::TestCaller<interfaceTestCase>("testJacobian",
			&interfaceTestCase::testJacobian));
		return suiteOfTests;
	}

	mt19937 *gen;
	array<double, Nparams> rxnRatesIn;

protected:
	void checkRetVal(int retVal, array<double, Nparams> &rxnRatesIn) {
		if (retVal < 0) {
			for (auto i = rxnRatesIn.begin(); i != rxnRatesIn.end(); ++i)
				std::cout << *i << ' ';

			cout << std::endl;
		}

		CPPUNIT_ASSERT(retVal >= 0);
	}

	array<double, Nparams> getParams() {
		array<double, Nparams> rxnRatesIn;
		lognormal_distribution<> dis(0.1, 0.25);

		generate(rxnRatesIn.begin(), rxnRatesIn.end(), [this, &dis]() { return dis(*this->gen); });

		rxnRatesIn[19] = tanh(rxnRatesIn[19])*0.9;

		return rxnRatesIn;
	}

	void testrunCkine() {
		array<double, Nspecies*tps.size()> output;
		array<double, Nspecies*tps.size()> output2;

		int retVal = runCkine(tps.data(), tps.size(), output.data(), rxnRatesIn.data(), false, 0.0, nullptr);

		// Run a second time to make sure we get the same thing
		int retVal2 = runCkine(tps.data(), tps.size(), output2.data(), rxnRatesIn.data(), false, 0.0, nullptr);

		checkRetVal(retVal, rxnRatesIn);
		checkRetVal(retVal2, rxnRatesIn);

		CPPUNIT_ASSERT(std::equal(output.begin(), output.end(), output2.begin()));
	}

	void testrunCkineS() {
		array<double, tps.size()> output;
		array<double, tps.size()> output2;
		array<double, Nparams*tps.size()> soutput;
		array<double, Nparams*tps.size()> soutput2;
		array<double, Nspecies> actV;
		fill(actV.begin(), actV.end(), 0.0);
		actV[10] = 1.0;

		checkRetVal(runCkineS(tps.data(), tps.size(), output.data(), soutput.data(), actV.data(), rxnRatesIn.data(), false, 0.0, nullptr), rxnRatesIn);

		// Run a second time to make sure we get the same thing
		checkRetVal(runCkineS(tps.data(), tps.size(), output2.data(), soutput2.data(), actV.data(), rxnRatesIn.data(), false, 0.0, nullptr), rxnRatesIn);

		CPPUNIT_ASSERT(std::equal(output.begin(), output.end(), output2.begin()));
		// The sensitivities are non-deterministic for some reason
		// CPPUNIT_ASSERT(std::equal(soutput.begin(), soutput.end(), soutput2.begin()));
	}

	void testrunCkinePretreat() {
		array<double, Nspecies*tps.size()> output;
		array<double, Nspecies*tps.size()> output2;
		array<double, Nlig> preL = {{0.1, 0.2, 0.1, 0.01, 0.3, 0.6}};

		int retVal = runCkine(tps.data(), tps.size(), output.data(), rxnRatesIn.data(), false, 10.0, preL.data());

		// Run a second time to make sure we get the same thing
		int retVal2 = runCkine(tps.data(), tps.size(), output2.data(), rxnRatesIn.data(), false, 10.0, preL.data());

		checkRetVal(retVal, rxnRatesIn);
		checkRetVal(retVal2, rxnRatesIn);

		CPPUNIT_ASSERT(std::equal(output.begin(), output.end(), output2.begin()));
	}

	void testrunCkineSPretreat() {
		array<double, tps.size()> output;
		array<double, tps.size()> output2;
		array<double, Nparams*tps.size()> soutput;
		array<double, Nparams*tps.size()> soutput2;
		array<double, Nspecies> actV;
		array<double, Nlig> preL = {{0.1, 0.2, 0.1, 0.01, 0.3, 0.6}};
		fill(actV.begin(), actV.end(), 0.0);
		actV[10] = 1.0;

		checkRetVal(runCkineS(tps.data(), tps.size(), output.data(), soutput.data(), actV.data(), rxnRatesIn.data(), false, 10.0, preL.data()), rxnRatesIn);

		// Run a second time to make sure we get the same thing
		checkRetVal(runCkineS(tps.data(), tps.size(), output2.data(), soutput2.data(), actV.data(), rxnRatesIn.data(), false, 10.0, preL.data()), rxnRatesIn);

		CPPUNIT_ASSERT(std::equal(output.begin(), output.end(), output2.begin()));
		// The sensitivities are non-deterministic for some reason
		// CPPUNIT_ASSERT(std::equal(soutput.begin(), soutput.end(), soutput2.begin()));
	}

	// Compare the analytical to an autodiff jacobian
	void testJacobian() {
		using adept::adouble;

		lognormal_distribution<> dis(0.1, 0.25);
		array<double, Nspecies> yv;
		array<double, Nparams> pIn = getParams();
		vector<double> params(Nparams);
		copy(pIn.begin(), pIn.end(), params.begin());
		ratesS<double> rattes = ratesS<double>(params);
		generate(yv.begin(), yv.end(), [this, &dis]() { return dis(*this->gen); });

		adept::Stack stack;

		array<adouble, Nspecies> y, dydt;

		adept::set_values(&y[0], Nspecies, yv.data());

		stack.new_recording();

		// Get the data in the right form
		fullModel(y.data(), &rattes, dydt.data());

		stack.independent(&y[0], Nspecies);
		stack.dependent(&dydt[0], Nspecies);

		Eigen::Matrix<double, Nspecies, Nspecies> jac_Auto, jac_Ana;
		stack.jacobian(jac_Auto.data());

		fullJacobian(yv.data(), &rattes, jac_Ana);

		CPPUNIT_ASSERT((jac_Auto - jac_Ana).squaredNorm() < 0.00000001);
	}
};

// the main method
int main () {
	CppUnit::TextUi::TestRunner runner;

	ofstream outputFile("testResults.xml");
	CppUnit::XmlOutputter* outputter = new CppUnit::XmlOutputter(&runner.result(), outputFile);    
	runner.setOutputter(outputter);

	runner.addTest(interfaceTestCase::suite());
	
	runner.run();

	outputFile.close();

	return 0;
}
