#ifndef CNN_CHANNEL_EXTRACTOR_GPU_HPP
#define CNN_CHANNEL_EXTRACTOR_GPU_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>
#include <kkl/cvk/icf_channel_extractor.hpp>

namespace cvk {

/**
 * @brief Convolution features extractor
 */
template<int conv_t1_size, int conv_t2_size>
class CNNChannelExtractorGPU : public ChannelExtractor {
public:

  /**
   * @brief constructor
   * @param params_dir  directory where the network parameters are contained
   */
  CNNChannelExtractorGPU(const std::string& params_dir)
  {}
  virtual ~CNNChannelExtractorGPU() override {}

  int numChannels() const override { return conv_t2_size; }

  std::vector<std::string> channelNames() const override {
    std::vector<std::string> names(conv_t2_size);
    for(int i=0; i<conv_t2_size; i++) {
      names[i] = (boost::format("layer%d") % i).str();
    }
    return names;
  }

  /**
   * @brief extract feature maps
   * @param bgr   BGR image
   * @param gray  gray image
   * @return extracted feature maps
   */
  std::vector<cv::Mat> extract(const cv::Mat& bgr, const cv::Mat& gray) const override {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, CV_BGR2RGB);

    std::vector<cv::Mat> layers;
    // TODO: extract features from the input image, and push them in layers

    for(int i=0; i<layers.size(); i++) {
      layers[i].convertTo(layers[i], CV_8UC3, 128.0);
    }

    return layers;
  }

private:

};

}


#endif // CNN_CHANNEL_EXTRACTOR_GPU_HPP
