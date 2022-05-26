#include "mlp.hpp"
#include <boost/numeric/ublas/operation.hpp>

MLP::MLP(const std::vector<int>& topology) {
    const auto random_01 = []() { return (double)rand() / RAND_MAX; };
    for (std::size_t i = 0; i < topology.size() - 1; i++) {
        Matrix weight_m(topology[i], topology[i + 1]);
        Vector bias_v(topology[i + 1]);

        std::generate(weight_m.begin1(), weight_m.end1(), random_01);
        std::generate(bias_v.begin(), bias_v.end(), random_01);

        this->weights.push_back(weight_m);
        this->biases.push_back(bias_v);
    }
}

double MLP::activationFunction(double x) {
    return tanh(x);
}

double MLP::activationFunctionDerivative(double x) {
    return 1 - tanhf(x) * tanhf(x);
}

MLP::Vector MLP::forward(Vector x) {
    const auto f = [this](double x) { return this->activationFunction(x); };
    for (std::size_t i = 0; i < this->weights.size(); i++) {
        Vector nx = biases[i];
        boost::numeric::ublas::axpy_prod(x, weights[i], nx, false);
        std::transform(nx.begin(), nx.end(), nx.begin(), f);
        x = std::move(nx);
    }
    return x;
}