#ifndef MULTILAYER_PERCEPTRON_MLP_MLP_HPP
#define MULTILAYER_PERCEPTRON_MLP_MLP_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <utility>

typedef std::vector<int> VInt;

class MLP {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;
    using Vector = boost::numeric::ublas::vector<double>;

public:
    explicit MLP(const std::vector<int>& topology);
    Vector forward(Vector input);
    std::pair<std::vector<Matrix>, std::vector<Vector>> backward(const Vector& expected);

private:
    std::vector<Matrix> weights;
    std::vector<Vector> biases;
    std::vector<Vector> outs;
    std::vector<Vector> nets;
    std::vector<Vector> deltas;

    double activationFunction(double x);
    double activationFunctionDerivative(double x);
};


#endif  // !MULTILAYER_PERCEPTRON_MLP_MLP_HPP