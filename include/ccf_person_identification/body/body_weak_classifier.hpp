#ifndef BODY_WEAK_CLASSIFIER_HPP
#define BODY_WEAK_CLASSIFIER_HPP

#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

#include <kkl/ml/online_boosting.hpp>
#include <kkl/ml/incremental_naive_bayes.hpp>

#include <kkl/cvk/icf_channel_bank.hpp>
#include <kkl/cvk/icf_channel_extractor.hpp>
#include <kkl/cvk/icf_integral_filter.hpp>
#include <ccf_person_identification/body/body_features.hpp>


namespace ccf_person_classifier {

/**
 * @brief Weak classifier for person body classification
 */
class BodyWeakClassifier : public kkl::ml::WeakClassifier<BodyFeatures::Ptr> {
public:
  /**
   * @brief constructor
   * @param channel_num  the channel number which indicates the channel to be used for feature extraction
   * @param filter       a filter which extracts the average in a certain area from the integral images
   */
  BodyWeakClassifier(int channel_num, const std::shared_ptr<cvk::IntegralFilter>& filter)
    : channel_num(channel_num),
      filter(filter),
      naive_bayes(new kkl::ml::IncrementalNaiveBayes())
  {}
  virtual ~BodyWeakClassifier() override {}

  /**
   * @brief update the classifier with a new training data
   * @param label     the label of the input
   * @param features  the feature maps
   */
  void update(double label, const BodyFeatures::Ptr& features) override {
    double x = filter->filter(features->integral_maps[channel_num]);
    naive_bayes->add(label, x);
  }

  /**
   * @brief predict the responce
   * @param features    the feature maps
   * @return            the response
   */
  double predict(const BodyFeatures::Ptr& features) const override {
    double x = filter->filter(features->integral_maps[channel_num]);
    return naive_bayes->predict_real(x) > 0.0 ? 1.0 : -1.0;
  }

  /**
   * @brief returns a string which describes the feature extraction parameters
   * @return a string which describes the feature extraction parameters
   */
  std::string toString() const override {
    std::stringstream sst;
    sst << "ch" << channel_num << " " << filter->tl().x << " " << filter->tl().y << " " << filter->size().width << " " << filter->size().height;
    return sst.str();
  }

  int channelNum() const {
    return channel_num;
  }

  const cv::Point2f& tl() const {
    return filter->tl();
  }

  const cv::Size2f& size() const {
    return filter->size();
  }

  cv::Rect calcRect(const cv::Size& size) const {
    return filter->calcRect(size);
  }

private:
  int channel_num;
  std::shared_ptr<cvk::IntegralFilter> filter;
  std::unique_ptr<kkl::ml::IncrementalNaiveBayes> naive_bayes;
};


/**
 * @brief BodyWeakClassifierGenerator
 */
class BodyWeakClassifierGenerator : public kkl::ml::WeakClassifierGenerator<BodyFeatures::Ptr> {
public:
  /**
   * @brief BodyWeakClassifierGenerator
   * @param num_channels
   * @param minimum_size
   * @param maximum_size
   */
  BodyWeakClassifierGenerator(int num_channels, double minimum_size = 0.2, double maximum_size = 0.5)
    : num_channels(num_channels),
      minimum_size(minimum_size),
      maximum_size(maximum_size),
      uniform_dist(0.0, 1.0)
  {
    assert(minimum_size > 0.0 && minimum_size <= 1.0);
    assert(minimum_size <= maximum_size && maximum_size > 0.0 && maximum_size <= 1.0);
  }
  virtual ~BodyWeakClassifierGenerator() override {}

  WeakClassifierPtr generate(const std::vector<double>& labels, const std::vector<Input>& samples) override {
    cv::Size2f size(uniform_dist(mt) * (maximum_size - minimum_size) + minimum_size, uniform_dist(mt) * (maximum_size - minimum_size) + minimum_size);
    cv::Point2f tl(uniform_dist(mt) * (1 - size.width), uniform_dist(mt) * (1 - size.height));

    int channel = std::uniform_int_distribution<>(0, num_channels - 1)(mt);
    std::shared_ptr<cvk::IntegralFilter> filter(new cvk::IntegralFilter(tl, size));

    WeakClassifierPtr classifier(new BodyWeakClassifier(channel, filter));
    for(int i=0; i<labels.size(); i++) {
      classifier->update(labels[i], samples[i]);
    }

    return classifier;
  }

private:
  const int num_channels;
  const double minimum_size;
  const double maximum_size;

  std::mt19937 mt;
  std::uniform_real_distribution<> uniform_dist;
};


}

#endif // BODY_WEAK_CLASSIFIER_HPP
