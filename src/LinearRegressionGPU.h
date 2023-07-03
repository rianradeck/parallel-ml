#include <vector>
#include <cassert>
#include <omp.h>
#include "MatrixGPU.h"

#define MULTOPERATOR %

struct LinearRegressionGPU
{
	double learningRate = 0.01;
	double bias = 0;
	MatrixGPU weights;
	int maxIter = 10000;

	LinearRegressionGPU()
	{
		weights = MatrixGPU(1, 1);
	}
	
	std::vector<double> predict(const MatrixGPU &X)
	{
		assert(X.col == weights.row);
		MatrixGPU ret2 = X MULTOPERATOR weights;
		std::vector<double> ret(X.row);
		for(size_t i = 0; i < X.row; ++i)
			ret[i] = ret2.getElement(i, 0) + bias;
		return ret;
	}

	double meanSquaredError(const MatrixGPU &X, const std::vector<double> &y)
	{
		assert(X.row == y.size());
		double ret = 0;
		std::vector<double> predicted = predict(X);
		for(std::size_t i = 0; i < X.row; ++i)
		{
			ret += (predicted[i] - y[i]) * (predicted[i] - y[i]);
		}
		ret /= X.row;
		return ret;
	}

	std::vector<double> computeGradient(const MatrixGPU &X, const std::vector<double> &y)
	{
		std::vector<double> ret(weights.row + 1);	
		std::vector<double> predicted = predict(X);
		for(std::size_t i = 0; i < X.row; ++i)
			ret[0] += 2.0 / X.row * (predicted[i] - y[i]);
		
		MatrixGPU aux(1, predicted.size());
		for(std::size_t i = 0; i < aux.col; ++i)
			aux.setElement(0, i, predicted[i] - y[i]);
			
		MatrixGPU grad = aux MULTOPERATOR X;

		#pragma omp parallel for
		for(std::size_t i = 1; i < ret.size(); ++i)
			ret[i] = grad.getElement(0, i - 1) * 2.0 / X.row;

		return ret;
	}
	
	//X is a n-d matrix with features
	//n samples, and each sample is a d-dimensional point
	//y is the labels 
	void fit(const MatrixGPU &X, const std::vector<double> &y)
	{
		assert(X.row == y.size());

		weights = MatrixGPU(X.col, 1);

		for(int currentIter = 0; currentIter < maxIter; currentIter++)
		{
			std::vector<double> gradient = computeGradient(X, y);
			bias -= learningRate * gradient[0];
			#pragma omp parallel for
			for(std::size_t i = 0; i < weights.row; ++i)
			{
				double aux = weights.getElement(i, 0);
				weights.setElement(i, 0, aux - learningRate * gradient[i + 1]);
			}
		}
	}
};

