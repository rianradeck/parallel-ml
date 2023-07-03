#include "LinearRegression.h"
#include "LinearRegressionGPU.h"
#include <random>
#include <iostream>
#include <chrono>

auto tStart = std::chrono::high_resolution_clock::now();
auto tEnd = std::chrono::high_resolution_clock::now();
auto duration_ = tEnd - tStart;

#define MEASURE_TIME_START tStart = std::chrono::high_resolution_clock::now();
#define MEASURE_TIME_END tEnd = std::chrono::high_resolution_clock::now(); duration_ = tEnd - tStart; std::cerr << "Measured time in seconds: " << duration_.count() / 1e9 << "\n";
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

	void genDataset(size_t n, size_t d, MatrixGPU &outX, std::vector<double> &outY, std::vector<double> &outWeights, double noiseFactor = 0)
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
		LinearRegressionGPU lGPU;
		MatrixGPU XGPU(3, 2);
		XGPU.setElement(0, 0, 1), XGPU.setElement(0, 1, 1);
		XGPU.setElement(1, 0, 1), XGPU.setElement(1, 1, 2);
		XGPU.setElement(2, 0, 2), XGPU.setElement(2, 1, 1);
		std::vector<double> y = {6, 9, 8};
		std::vector<double> yGPU = {6, 9, 8};
		std::cerr << "--- CPU start ---\n";
		MEASURE_TIME_START
		l.fit(X, y);
		double mse = l.meanSquaredError(X, y);
		MEASURE_TIME_END
		std::cerr << "--- GPU start ---\n";
		MEASURE_TIME_START
		lGPU.fit(XGPU, yGPU);
		double mseGPU = lGPU.meanSquaredError(XGPU, yGPU);
		MEASURE_TIME_END
		return mse < 1e-2 and mseGPU < 1e-2;
	}
	bool test2()
	{
		LinearRegression l;
		Matrix X(3, 3);
		X.setElement(0, 0, 1), X.setElement(0, 1, 1), X.setElement(0, 2, 1);
		X.setElement(1, 0, 1), X.setElement(1, 1, 2), X.setElement(1, 2, 3);
		X.setElement(2, 0, 2), X.setElement(2, 1, 1), X.setElement(2, 2, 10);
		LinearRegressionGPU lGPU;
		MatrixGPU XGPU(3, 2);
		XGPU.setElement(0, 0, 1), XGPU.setElement(0, 1, 1), XGPU.setElement(0, 2, 1);
		XGPU.setElement(1, 0, 1), XGPU.setElement(1, 1, 2), XGPU.setElement(1, 2, 3);
		XGPU.setElement(2, 0, 2), XGPU.setElement(2, 1, 1), XGPU.setElement(2, 2, 10);
		std::vector<double> y = {10, 21, 48};
		std::vector<double> yGPU = {10, 21, 48};
		std::cerr << "--- CPU start ---\n";
		MEASURE_TIME_START
		l.fit(X, y);
		double mse = l.meanSquaredError(X, y);
		MEASURE_TIME_END
		std::cerr << "--- GPU start ---\n";
		MEASURE_TIME_START
		lGPU.fit(XGPU, yGPU);
		double mseGPU = lGPU.meanSquaredError(XGPU, yGPU);
		MEASURE_TIME_END
		return mse < 1e-2 and mseGPU < 1e-2;

	}
	bool test3()
	{
		LinearRegression l;
		Matrix X(20, 5);
		LinearRegressionGPU lGPU;
		MatrixGPU XGPU(20, 5);
		std::vector<double> y, weights;
		std::vector<double> yGPU, weightsGPU;
		genDataset(20, 5, X, y, weights);
		genDataset(20, 5, XGPU, yGPU, weightsGPU);

		std::cerr << "--- CPU start ---\n";
		MEASURE_TIME_START
		l.fit(X, y);
		double mse = l.meanSquaredError(X, y);
		MEASURE_TIME_END
		std::cerr << "--- GPU start ---\n";
		MEASURE_TIME_START
		lGPU.fit(XGPU, yGPU);
		double mseGPU = lGPU.meanSquaredError(XGPU, yGPU);
		MEASURE_TIME_END
		return mse < 1e-2 and mseGPU < 1e-2;

	}
	bool test4()
	{
		LinearRegression l;
		Matrix X(100, 10);
		LinearRegressionGPU lGPU;
		MatrixGPU XGPU(100, 10);
		std::vector<double> y, weights;
		std::vector<double> yGPU, weightsGPU;
		genDataset(100, 10, X, y, weights);
		genDataset(100, 10, XGPU, yGPU, weightsGPU);

		std::cerr << "--- CPU start ---\n";
		MEASURE_TIME_START
		l.fit(X, y);
		double mse = l.meanSquaredError(X, y);
		MEASURE_TIME_END
		std::cerr << "--- GPU start ---\n";
		MEASURE_TIME_START
		lGPU.fit(XGPU, yGPU);
		double mseGPU = lGPU.meanSquaredError(XGPU, yGPU);
		MEASURE_TIME_END
		return mse < 1e-2 and mseGPU < 1e-2;
	}
	bool testBig()
	{	
		LinearRegression l;
		Matrix X(10000, 100);
		LinearRegressionGPU lGPU;
		MatrixGPU XGPU(10000, 100);
		std::vector<double> y, weights;
		std::vector<double> yGPU, weightsGPU;
		genDataset(10000, 100, X, y, weights);
		genDataset(10000, 100, XGPU, yGPU, weightsGPU);

		std::cerr << "--- CPU start ---\n";
		MEASURE_TIME_START
		l.fit(X, y);
		double mse = l.meanSquaredError(X, y);
		MEASURE_TIME_END
		std::cerr << "--- GPU start ---\n";
		MEASURE_TIME_START
		lGPU.fit(XGPU, yGPU);
		double mseGPU = lGPU.meanSquaredError(XGPU, yGPU);
		MEASURE_TIME_END
		return mse < 1e-2 and mseGPU < 1e-2;
	}
#define TEST(X) if(X()) \
std::cerr << #X << ": \033[32;1;4mPASS\033[0m" << std::endl; \
else \
std::cerr << #X << ": \033[31;1;4mFAIL\033[0m" << std::endl; \

	void testLinearRegression()
	{	
		// auto t = std::chrono::high_resolution_clock::now();
		TEST(test1);
		// auto t1 = std::chrono::high_resolution_clock::now();
		// auto d1 = t1 - t;
		// std::cerr << d1.count() / 1e9 << std::endl;
		TEST(test2);
		// auto t2 = std::chrono::high_resolution_clock::now();
		// auto d2 = t2 - t1;
		// std::cerr << d2.count() / 1e9 << std::endl;
		TEST(test3);
		// auto t3 = std::chrono::high_resolution_clock::now();
		// auto d3 = t3 - t2;
		// std::cerr << d3.count() / 1e9 << std::endl;
		TEST(test4);
		// auto t4 = std::chrono::high_resolution_clock::now();
		// auto d4 = t4 - t3;
		// std::cerr << d4.count() / 1e9 << std::endl;
		TEST(testBig);
		// auto t5 = std::chrono::high_resolution_clock::now();
		// auto d5 = t5 - t4;
		// std::cerr << d5.count() / 1e9 << std::endl;
	}
#undef TEST
};

