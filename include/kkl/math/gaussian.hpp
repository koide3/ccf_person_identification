/**
* gaussian.hpp
* @author : koide
**/
#ifndef KKL_GAUSSIAN_HPP
#define KKL_GAUSSIAN_HPP


#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace kkl{
namespace math{


/**
 * @brief Gaussian distribution estimater based on kalman filter
 * @ref "On-line boosting and vision"
 */
class GaussianEstimater {
public:
    /**
     * @brief GaussianEstimater
     * @param init_mean  initial mean
     * @param init_var   initial variance
     * @param init_P     variance of the initial distribution
     */
    GaussianEstimater(double init_mean = 0.0, double init_var = 1.0, double init_P = 1000.0)
        : P(init_P), mean(init_mean), var(init_var)
    {}

    /**
     * @brief update
     * @param w  weight
     * @param f  input
     */
    void update(double w, double f) {
        const double R = 0.01;
        double K = std::min(1.0 - 1e-6, w * P / (P + R));
        mean = K * f + (1 - K) * mean;
        var = K * (f - mean) * (f - mean) + (1 - K) * var;
        P = (1 - K) * P;
    }

    /**
     * @brief probabilistic density function
     * @param f   input
     * @return PDF
     */
    double prob(double f) const {
        return 1.0 / sqrt(2 * M_PI * var) * exp(-(f - mean) * (f - mean) / (2 * var));
    }

    /**
     * @brief probabilistic density function
     * @param f  input
     * @return PDF
     */
    double operator()(double f) const {
        return prob(f);
    }

public:
    double P;
    double mean;
    double var;
};


}
}

#endif
