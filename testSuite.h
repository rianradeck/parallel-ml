#include "LinearRegression.h"
#include <random>
#include <iostream>

namespace testSuite
{
	void genDataset(size_t n, size_t d, Matrix &outX, std::vector<double> &outY, std::vector<double> &outWeights, double noiseFactor = 0)
	{
		std::random_device rd;
		std::mt19937 pgen(rd());
		std::uniform_real_distribution<> dist(0, 1);
		std::normal_distribution<> noiseDist(0, noiseFactor);
		outY.resize(n);
		outWeights.resize(d + 1);
		for(size_t i = 0; i < d + 1; ++i)
			outWeights[i] = dist(pgen);
		
		for(size_t i = 0; i < n; ++i)
		{
			double acc = outWeights[0];
			for(size_t j = 0; j < d; ++j)
			{
				outX.setElement(i, j, dist(pgen));
				acc += outX.getElement(i, j) * outWeights[j + 1];
			}
			outY[i] = acc + noiseDist(pgen);
		}
	}

	bool test1()
	{
		LinearRegression l;
		Matrix X(3, 2);
		X.setElement(0, 0, 1), X.setElement(0, 1, 1);
		X.setElement(1, 0, 1), X.setElement(1, 1, 2);
		X.setElement(2, 0, 2), X.setElement(2, 1, 1);
		std::vector<double> y = {6, 9, 8};
		l.fit(X, y);
		double mse = l.meanSquaredError(X, y);
		return mse < 1e-2;
	}
	bool test2()
	{
		LinearRegression l;
		Matrix X(3, 3);
		X.setElement(0, 0, 1), X.setElement(0, 1, 1), X.setElement(0, 2, 1);
		X.setElement(1, 0, 1), X.setElement(1, 1, 2), X.setElement(1, 2, 3);
		X.setElement(2, 0, 2), X.setElement(2, 1, 1), X.setElement(2, 2, 10);
		std::vector<double> y = {10, 21, 48};
		l.fit(X, y);
		double mse = l.meanSquaredError(X, y);
		return mse < 1e-2;

	}
	bool test3()
	{
		LinearRegression l;
		Matrix X(20, 5);
		std::vector<double> y, weights;
		genDataset(20, 5, X, y, weights);
		l.fit(X, y);
		double mse = l.meanSquaredError(X, y);
		return mse < 1e-2;

	}
	bool test4()
	{
		LinearRegression l;
		Matrix X(100, 10);
		std::vector<double> y, weights;
		genDataset(100, 10, X, y, weights);
		l.fit(X, y);
		double mse = l.meanSquaredError(X, y);
		return mse < 1e-2;
	}
	bool testBig()
	{
		LinearRegression l;
		Matrix X(1000, 10);
		std::vector<double> y, weights;
		genDataset(1000, 10, X, y, weights);
		l.fit(X, y);
		double mse = l.meanSquaredError(X, y);
		return mse < 1e-2;
	}
#define TEST(X) if(X()) \
std::cerr << #X << ": \033[32;1;4mPASS\033[0m" << std::endl; \
else \
std::cerr << #X << ": \033[31;1;4mFAIL\033[0m" << std::endl; \

	void testLinearRegression()
	{	
		TEST(test1);
		TEST(test2);
		TEST(test3);
		TEST(test4);
		TEST(testBig);
	}
#undef TEST
};

