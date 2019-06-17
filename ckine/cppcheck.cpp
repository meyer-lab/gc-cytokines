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
#include "reaction.hpp"

using namespace std;

constexpr array<double, 4> tps = {{0.1, 10.0, 1000.0, 10000.0}};


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

		// Check sensitivities reproducibility
		for (size_t ii = 0; ii < soutput.size(); ii++) {
			CPPUNIT_ASSERT_EQUAL_MESSAGE(std::string("runCkineS sensitivity at pos ") + std::to_string(ii), soutput[ii], soutput2[ii]);
		}
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

		// Check sensitivities reproducibility
		for (size_t ii = 0; ii < soutput.size(); ii++) {
			CPPUNIT_ASSERT_EQUAL_MESSAGE(std::string("Pretreat sensitivity at pos ") + std::to_string(ii), soutput[ii], soutput2[ii]);
		}
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
