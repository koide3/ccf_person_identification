#ifndef KKL_ICF_CHANNEL_BANK_HPP
#define KKL_ICF_CHANNEL_BANK_HPP

#include <memory>
#include <numeric>
#include <iterator>
#include <opencv2/opencv.hpp>

#include <kkl/cvk/icf_channel_extractor.hpp>

namespace cvk {

/**
 * @brief Channel bank of ICF
 */
class ChannelBank : public ChannelExtractor {
public:
  ChannelBank() {}
  ~ChannelBank() {}

  void addExtractor(const std::shared_ptr<ChannelExtractor>& extractor) {
    extractors.push_back(extractor);
  }

  int numChannels() const override {
    return std::accumulate(extractors.begin(), extractors.end(), 0, [=](int n, const std::shared_ptr<ChannelExtractor>& e) { return n + e->numChannels(); });
  }

  std::vector<std::string> channelNames() const override {
    std::vector<std::string> names;
    for(const auto& e : extractors) {
      auto n = e->channelNames();
      std::copy(n.begin(), n.end(), std::back_inserter(names));
    }
    return names;
  }

  std::vector<cv::Mat> extract(const cv::Mat& rgb, const cv::Mat& gray) const override {
    std::vector<cv::Mat> channels;
    channels.reserve(numChannels());

    for(const auto& extractor : extractors) {
      auto chs = extractor->extract(rgb, gray);
      std::copy(chs.begin(), chs.end(), std::back_inserter(channels));
    }

    return channels;
  }

  std::vector<cv::Mat> extract(const cv::Mat& rgb) const {
    cv::Mat gray;
    cv::cvtColor(rgb, gray, CV_BGR2GRAY);

    return extract(rgb, gray);
  }

private:
  std::vector<std::shared_ptr<ChannelExtractor>> extractors;
};

}

#endif // CHANNEL_BANK_HPP
