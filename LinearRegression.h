#include <vector>
#include <cassert>

#include "Matrix.h"

struct LinearRegression
{
	double learningRate = 0.01;
	std::vector<double> weights;
	int maxIter = 100000;
	
	std::vector<double> predict(const Matrix &X)
	{
		assert(X.col + 1 == weights.size());
		std::vector<double> ret(X.row);
		for(size_t i = 0; i < ret.size(); ++i)
		{
			double acc = weights[0];
			for(std::size_t j = 1; j < weights.size(); ++j)
				acc += weights[j] * X.getElement(i, j - 1);
			ret[i] = acc;
		}
		return ret;
	}

	double meanSquaredError(const Matrix &X, const std::vector<double> &y)
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

	std::vector<double> computeGradient(const Matrix &X, const std::vector<double> &y)
	{
		std::vector<double> ret(weights.size());	
		std::vector<double> predicted = predict(X);
		for(std::size_t i = 0; i < X.row; ++i)
		{
			ret[0] += 2.0 / X.row * (predicted[i] - y[i]);
		}
			
		for(std::size_t i = 1; i < ret.size(); ++i)
		{
			for(std::size_t j = 0; j < X.row; ++j)
			{
				ret[i] += 2.0 / X.row * X.getElement(j, i - 1) * (predicted[j] - y[j]);
			}
		}

		return ret;
	}
	
	//X is a n-d matrix with features
	//n samples, and each sample is a d-dimensional point
	//y is the labels 
	void fit(const Matrix &X, const std::vector<double> &y)
	{
		assert(X.row == y.size());

		weights.resize(X.col + 1);

		for(int currentIter = 0; currentIter < maxIter; currentIter++)
		{
			std::vector<double> gradient = computeGradient(X, y);
			for(std::size_t i = 0; i < weights.size(); ++i)
				weights[i] -= learningRate * gradient[i];
		}
	}
};

