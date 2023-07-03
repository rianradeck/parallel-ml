#include <vector>
#include <cassert>

#include "Matrix.h"

struct LinearRegression
{
	double learningRate = 0.01;
	double bias = 0;
	Matrix weights;
	int maxIter = 4000;

	LinearRegression()
	{
		weights = Matrix(1, 1);
	}
	
	std::vector<double> predict(const Matrix &X)
	{
		assert(X.col == weights.row);
		Matrix ret2 = X % weights;
		std::vector<double> ret(X.row);
		for(size_t i = 0; i < X.row; ++i)
			ret[i] = ret2.getElement(i, 0) + bias;
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
		std::vector<double> ret(weights.row + 1);	
		std::vector<double> predicted = predict(X);
		for(std::size_t i = 0; i < X.row; ++i)
		{
			ret[0] += 2.0 / X.row * (predicted[i] - y[i]);
		}
		
		Matrix aux(1, predicted.size());
		for(std::size_t i = 0; i < aux.col; ++i)
			aux.setElement(0, i, predicted[i] - y[i]);
			
		Matrix grad = aux % X;

		for(std::size_t i = 1; i < ret.size(); ++i)
			ret[i] = grad.getElement(0, i - 1) * 2.0 / X.row;

		return ret;
	}
	
	//X is a n-d matrix with features
	//n samples, and each sample is a d-dimensional point
	//y is the labels 
	void fit(const Matrix &X, const std::vector<double> &y)
	{
		assert(X.row == y.size());

		weights = Matrix(X.col, 1);

		for(int currentIter = 0; currentIter < maxIter; currentIter++)
		{
			std::vector<double> gradient = computeGradient(X, y);
			bias -= learningRate * gradient[0];
			for(std::size_t i = 0; i < weights.row; ++i)
			{
				double aux = weights.getElement(i, 0);
				weights.setElement(i, 0, aux - learningRate * gradient[i + 1]);
			}
		}

	}
};

