#ifndef KKL_ICF_CHANNEL_EXTRACTOR_HPP
#define KKL_ICF_CHANNEL_EXTRACTOR_HPP

#include <boost/format.hpp>
#include <opencv2/opencv.hpp>

namespace cvk {

/**
 * @brief Channel extractor base class
 */
class ChannelExtractor {
public:
  // constructor, destructor
  ChannelExtractor() {}
  virtual ~ChannelExtractor() {}

  /**
   * @brief the number of extracted channels
   * @return the number of extracted channels
   */
  virtual int numChannels() const = 0;

  /**
   * @brief the names of extracted channels
   * @return the names of extracted channels
   */
  virtual std::vector<std::string> channelNames() const = 0;

  /**
   * @brief extract channels
   * @param rgb   rgb image
   * @param gray  grayscale image
   * @return extracted channels
   */
  virtual std::vector<cv::Mat> extract(const cv::Mat& rgb, const cv::Mat& gray) const = 0;
};

/**
 * @brief HSV channels extractor
 */
class ChannelExtractorHSV : public ChannelExtractor {
public:
  ChannelExtractorHSV() {}
  ~ChannelExtractorHSV() {}

  int numChannels() const override { return 3; }

  std::vector<std::string> channelNames() const override {
    return std::vector<std::string> {
      "hue", "sat", "val"
    };
  }

  std::vector<cv::Mat> extract(const cv::Mat& rgb, const cv::Mat& gray) const override {
    cv::Mat hsv;
    cv::cvtColor(rgb, hsv, CV_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    return channels;
  }
};

/**
 * @brief LUV channels extractor
 */
class ChannelExtractorLUV: public ChannelExtractor {
public:
  ChannelExtractorLUV() {}
  ~ChannelExtractorLUV() {}

  int numChannels() const override { return 3; }

  std::vector<std::string> channelNames() const override {
    return std::vector<std::string> {
      "l", "u", "v"
    };
  }

  std::vector<cv::Mat> extract(const cv::Mat& rgb, const cv::Mat& gray) const override {
    cv::Mat hsv;
    cv::cvtColor(rgb, hsv, CV_BGR2Luv);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    return channels;
  }
};

/**
 * @brief Gradient channels extractor
 */
class ChannelExtractorGrads : public ChannelExtractor {
public:
  ChannelExtractorGrads(int n_bins = 6)
    : n_bins(n_bins)
  {}
  ~ChannelExtractorGrads() {}

  int numChannels() const override { return n_bins + 1; }

  std::vector<std::string> channelNames() const override {
    std::vector<std::string> names;
    for(int i=0; i<n_bins; i++) {
      names.push_back((boost::format("grad%d") % i).str());
    }
    names.push_back("grad");

    return names;
  }

  std::vector<cv::Mat> extract(const cv::Mat& rgb, const cv::Mat& gray) const override {
    cv::Mat x_edge, y_edge;
    cv::Sobel(gray, x_edge, CV_16S, 1, 0, 3);
    cv::Sobel(gray, y_edge, CV_16S, 0, 1, 3);

    x_edge.convertTo(x_edge, CV_32F);
    y_edge.convertTo(y_edge, CV_32F);

    cv::Mat magnitude, angle;
    cv::cartToPolar(x_edge, y_edge, magnitude, angle, false);

    angle.convertTo(angle, CV_16SC1, static_cast<double>(n_bins) / M_PI);

    std::vector<cv::Mat> channels(n_bins + 1);
    for(auto& channel : channels) {
      channel = cv::Mat(rgb.rows, rgb.cols, CV_8UC1, cv::Scalar::all(0));
    }

    std::vector<int> subs(n_bins * 2 + 1);
    for(int i=0; i<subs.size(); i++) {
      subs[i] = n_bins * (i / n_bins);
    }

    for(int i=0; i<rgb.rows * rgb.cols; i++) {
      int index = angle.at<short>(i);
      index -= subs[index];
      channels[index].at<uchar>(i) = magnitude.at<float>(i);
    }

    magnitude.convertTo(channels.back(), CV_8UC1);

    return channels;
  }

private:
  int n_bins;
};

}


#endif // CHANNEL_EXTRACTOR_HPP
