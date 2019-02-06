#ifndef KKL_INCREMENTAL_NAIVE_BAYES_HPP
#define KKL_INCREMENTAL_NAIVE_BAYES_HPP

#include <kkl/math/gaussian.hpp>

namespace kkl{
  namespace ml {

class IncrementalNaiveBayes {
public:
  IncrementalNaiveBayes()
    : pos_accum_w(1e-6),
      neg_accum_w(1e-6)
  {}

  void add(double label, double x) {
    if(label > 0.0) {
      pos_accum_w += label;
      pos_dist.update(label, x);
    } else {
      neg_accum_w += std::abs(label);
      neg_dist.update(std::abs(label), x);
    }
  }

  int predict(double x) const {
    return predict_real(x) > 0.0 ? 1 : -1;
  }

  double predict_real(double x) const {
    double sum_pos_neg = pos_accum_w + neg_accum_w;
    double pos_priori = pos_accum_w / sum_pos_neg;
    double neg_priori = neg_accum_w / sum_pos_neg;

    double pos_posterior = pos_priori * pos_dist(x);
    double neg_posterior = neg_priori * neg_dist(x);

    return pos_posterior - neg_posterior;
  }

private:
  double pos_accum_w;
  double neg_accum_w;
  kkl::math::GaussianEstimater pos_dist;
  kkl::math::GaussianEstimater neg_dist;
};

  }
}

#endif // NAIVE_BAYES_HPP
