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

MLP::Vector 
MLP::forward(Vector x) {
    const auto f = [this](double x) { return this->activationFunction(x); };
    for (std::size_t i = 0; i < this->weights.size(); i++) {
        Vector nx = biases[i];
        nx = boost::numeric::ublas::axpy_prod(x, weights[i], nx, false);
        nets.push_back(nx);
        std::transform(nx.begin(), nx.end(), nx.begin(), f);
        x = std::move(nx);
        outs.push_back(x);
    }
    return x;
}


std::pair<std::vector<MLP::Matrix>, std::vector<MLP::Vector>>
MLP::backward(const Vector& expected) {
    Matrix dL_dW(this->weights.back().size1(), 
                 this->weights.back().size2());

    const int k = outs.size() - 1;
    for (std::size_t i = 0; i < dL_dW.size1(); i++) {
        for (std::size_t j = 0; j < dL_dW.size2(); j++) {
            auto dL_dOut_j = 2.0 * (expected[j] - outs[k][j]);
            auto dOut_dW_ij = this->activationFunctionDerivative(nets[k][j]) * outs[k-1][i];
            dL_dW(i, j) = dL_dOut_j * dOut_dW_ij;
        }
    }
}