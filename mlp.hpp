#ifndef MULTILAYER_PERCEPTRON_MLP_MLP_HPP
#define MULTILAYER_PERCEPTRON_MLP_MLP_HPP

#include <boost/numeric/ublas/matrix.hpp>

class MLP {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;
    using Vector = boost::numeric::ublas::vector<double>;

public:
    explicit MLP(const std::vector<int>& topology);
    Vector forward(const Vector& input);

private:
    std::vector<Matrix> weights;
    std::vector<Vector> biases;

    double activationFunction(double x);
    double activationFunctionDerivative(double x);
};


#endif  // !MULTILAYER_PERCEPTRON_MLP_MLP_HPP