#include <vector>
#include <cassert>

struct LinearRegression
{
	double learningRate = 0.01;
	std::vector<double> weights;
	int maxIter = 100000;
	
	double predict(const std::vector<double> &x)
	{
		assert(x.size() + 1 == weights.size());
		double ret = weights[0];
		for(std::size_t i = 0; i < x.size(); ++i)
			ret += weights[i + 1] * x[i];
		return ret;
	}

	double meanSquaredError(const std::vector<std::vector<double>> &X, const std::vector<double> &y)
	{
		assert(X.size() == y.size());
		double ret = 0;
		for(std::size_t i = 0; i < X.size(); ++i)
		{
			double predicted = predict(X[i]);
			ret += (predicted - y[i]) * (predicted - y[i]);
		}
		ret /= X.size();
		return ret;
	}

	std::vector<double> computeGradient(const std::vector<std::vector<double>> &X, const std::vector<double> &y)
	{
		std::vector<double> ret(weights.size());	
		std::vector<double> predicted(X.size());
		for(std::size_t i = 0; i < X.size(); ++i)
		{
			predicted[i] = predict(X[i]);
			ret[0] += 2.0 / X.size() * (predicted[i] - y[i]);
		}
			
		for(std::size_t i = 1; i < ret.size(); ++i)
		{
			for(std::size_t j = 0; j < X.size(); ++j)
			{
				ret[i] += 2.0 / X.size() * X[j][i - 1] * (predicted[j] - y[j]);
			}
		}

		return ret;
	}
	
	//X is a n-d matrix with features
	//n samples, and each sample is a d-dimensional point
	//y is the labels 
	void fit(const std::vector<std::vector<double>> &X, const std::vector<double> &y)
	{
		assert(!X.empty());
		assert(X.size() == y.size());

		weights.resize(X[0].size() + 1);

		for(int currentIter = 0; currentIter < maxIter; currentIter++)
		{
			std::vector<double> gradient = computeGradient(X, y);
			for(std::size_t i = 0; i < weights.size(); ++i)
				weights[i] -= learningRate * gradient[i];
		}
	}
};

