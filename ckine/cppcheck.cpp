#include <iostream>
#include <string>
#include <random>
#include <array>
#include <random>
#include <algorithm>

#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/XmlOutputter.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestRunner.h>

#include "model.hpp"

using namespace std;

class interfaceTestCase : public CppUnit::TestCase {
public:
	/// Setup method
	void setUp() {
		random_device rd;
		gen = new mt19937(rd());
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

		return suiteOfTests;
	}

	mt19937 *gen;

protected:
	void testrunCkine() {
		uniform_real_distribution<> dis(0.0, 10.0);

		array<double, 7> tps = {0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0};
		array<double, 56*7> output;
		array<double, 56*7> output2;
		array<double, 15> rxnRatesIn;
		array<double, 11> trafRatesIn;

		for (size_t ii = 0; ii < 1000; ii++) {
			generate(rxnRatesIn.begin(), rxnRatesIn.end(), [this, &dis]() { return dis(*this->gen); });
			generate(trafRatesIn.begin(), trafRatesIn.end(), [this, &dis]() { return dis(*this->gen); });

			trafRatesIn[2] /= 10.0;

			int retVal = runCkine(tps.data(), tps.size(), output.data(), rxnRatesIn.data(), trafRatesIn.data());

			// Run a second time to make sure we get the same thing
			int retVal2 = runCkine(tps.data(), tps.size(), output2.data(), rxnRatesIn.data(), trafRatesIn.data());

			std::transform(output.begin(), output.end(), output2.begin(), output2.begin(), std::minus<double>());
			double sumDiff = inner_product(output2.begin(), output2.end(), output2.begin(), 0.0);

			if (retVal < 0) {
				for (auto i = rxnRatesIn.begin(); i != rxnRatesIn.end(); ++i)
					std::cout << *i << ' ';

				cout << std::endl;

				for (auto i = trafRatesIn.begin(); i != trafRatesIn.end(); ++i)
					std::cout << *i << ' ';

				cout << std::endl;
			}

			CPPUNIT_ASSERT(retVal >= 0);
			CPPUNIT_ASSERT(retVal2 >= 0);
			CPPUNIT_ASSERT(sumDiff == 0);
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
