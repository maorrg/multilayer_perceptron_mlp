#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <vector>

namespace ublas = boost::numeric::ublas;

class MLP {
private:
    using WeightMatrix = ublas::matrix<double>;
    using BiasVector = ublas::vector<double>;

    std::vector<WeightMatrix> weights;
    std::vector<BiasVector> biases;

public:
    explicit MLP(const std::vector<int>& topology) {
        for (std::size_t i = 0; i < topology.size() - 1; i++) {
            WeightMatrix weight_m(topology[i], topology[i + 1]);
            BiasVector bias_v(topology[i + 1]);
            
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

    float activationFunction(float x) {
            return tanhf(x);
    }

    float activationFunctionDerivative(float x) {
            return 1 - tanhf(x) * tanhf(x);
    }
    
    auto forward(const ublas::vector<double>& input) {
        ublas::vector<double> output = input;
        for (std::size_t i = 0; i < this->weights.size(); i++) {
            output = ublas::prod(output, this->weights[i]) + this->biases[i];
            std::transform(output.begin(), output.end(), output.begin(),
                           [this](double x) { return this->activationFunction(x); });
        }
        return output;
    }
};

int main () {
    std::vector<int> topology = {2, 3, 4, 1}; // {Input Layer, First Hidden Layer, ... , Output Layer}
    
    MLP mlp(topology);
    
    ublas::vector<double> input(2);
    input(0) = 0.5;
    input(1) = 0.7;

    ublas::vector<double> output = mlp.forward(input);
    for (auto x : output) {
        std::cout << x << std::endl;
    }
}