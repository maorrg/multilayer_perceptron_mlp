#include "mlp.hpp"


MLP::MLP(const std::vector<int>& topology) {
    for (std::size_t i = 0; i < topology.size() - 1; i++) {
        Matrix weight_m(topology[i], topology[i + 1]);
        Vector bias_v(topology[i + 1]);

        for (std::size_t j = 0; j < weight_m.size1(); j++) {
            for (std::size_t k = 0; k < weight_m.size2(); k++) {
                weight_m(j, k) = (double)rand() / RAND_MAX;
            }
        }
        for (std::size_t j = 0; j < bias_v.size(); j++) {
            bias_v(j) = (double)rand() / RAND_MAX;
        }

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

MLP::Vector MLP::forward(const Vector& input) {
    Vector output = input;
    for (std::size_t i = 0; i < this->weights.size(); i++) {
        output = boost::numeric::ublas::prod(output, this->weights[i]) + this->biases[i];
        std::transform(output.begin(), output.end(), output.begin(),
                       [this](double x) { return this->activationFunction(x); });
    }
    return output;
}