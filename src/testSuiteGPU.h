#include "LinearRegressionGPU.h"
#include <random>
#include <iostream>
#include <chrono>

auto tStart = std::chrono::high_resolution_clock::now();
auto tEnd = std::chrono::high_resolution_clock::now();
auto duration_ = tEnd - tStart;

#define MEASURE_TIME_START tStart = std::chrono::high_resolution_clock::now();
#define MEASURE_TIME_END tEnd = std::chrono::high_resolution_clock::now(); duration_ = tEnd - tStart; std::cerr << "Measured time in seconds: " << duration_.count() / 1e9 << "\n";

namespace testSuiteGPU
{

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

	bool test(int N, int D){
		LinearRegressionGPU lGPU;
		MatrixGPU XGPU(N, D);
		std::vector<double> yGPU, weightsGPU;
		genDataset(N, D, XGPU, yGPU, weightsGPU);

		std::cerr << "--- GPU start ---\n";
		std::cerr << N << " " << D << "\n";
		MEASURE_TIME_START
		lGPU.fit(XGPU, yGPU);
		double mseGPU = lGPU.meanSquaredError(XGPU, yGPU);
		MEASURE_TIME_END
		return (mseGPU < 1e-2);
	}

	bool test1()
	{
		LinearRegressionGPU lGPU;
		MatrixGPU XGPU(3, 2);
		XGPU.setElement(0, 0, 1), XGPU.setElement(0, 1, 1);
		XGPU.setElement(1, 0, 1), XGPU.setElement(1, 1, 2);
		XGPU.setElement(2, 0, 2), XGPU.setElement(2, 1, 1);
		std::vector<double> yGPU = {6, 9, 8};

		std::cerr << "--- GPU start ---\n";
		MEASURE_TIME_START
		lGPU.fit(XGPU, yGPU);
		double mseGPU = lGPU.meanSquaredError(XGPU, yGPU);
		MEASURE_TIME_END
		return (mseGPU < 1e-2);
	}
	bool test2()
	{
		LinearRegressionGPU lGPU;
		MatrixGPU XGPU(3, 2);
		XGPU.setElement(0, 0, 1), XGPU.setElement(0, 1, 1), XGPU.setElement(0, 2, 1);
		XGPU.setElement(1, 0, 1), XGPU.setElement(1, 1, 2), XGPU.setElement(1, 2, 3);
		XGPU.setElement(2, 0, 2), XGPU.setElement(2, 1, 1), XGPU.setElement(2, 2, 10);
		std::vector<double> yGPU = {10, 21, 48};

		std::cerr << "--- GPU start ---\n";
		MEASURE_TIME_START
		lGPU.fit(XGPU, yGPU);
		double mseGPU = lGPU.meanSquaredError(XGPU, yGPU);
		MEASURE_TIME_END
		return (mseGPU < 1e-2);

	}
#define TEST(X, a, b) if(X(a, b)) \
std::cerr << #X << ": \033[32;1;4mPASS\033[0m" << std::endl; \
else \
std::cerr << #X << ": \033[31;1;4mFAIL\033[0m" << std::endl; \

	void testLinearRegression()
	{	
		// TEST(test1);
		// TEST(test2);
		// TEST(test, 20, 5);
		// TEST(test, 100, 10);
		// TEST(test, 1000, 10);
		// TEST(test, 100, 100);
		// TEST(test, 1000, 100);
		// TEST(test, 10000, 100);
		TEST(test, 50000, 500);
	}
#undef TEST
};

