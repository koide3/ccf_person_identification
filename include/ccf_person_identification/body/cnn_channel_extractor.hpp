#ifndef CNN_CHANNEL_EXTRACTOR_HPP
#define CNN_CHANNEL_EXTRACTOR_HPP

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>
#include <kkl/cvk/icf_channel_extractor.hpp>

namespace cvk {

/**
 * @brief proxy class to access parameters of dlib's convolusion layers
 */
class ConvParamsProxy {
public:

  /**
   * @brief constructor
   */
  template<typename con_>
  ConvParamsProxy(con_& con)
    : params(con.get_layer_params())
  {
    int k = (params.size() - con.num_filters()) / (con.nr() * con.nc() * con.num_filters());
    filters_ = dlib::alias_tensor(con.num_filters(), k, con.nr(), con.nc());
    biases_ = dlib::alias_tensor(1, con.num_filters());
  }

  /**
   * @brief kernel of the convolution filter
   * @return convolution kernel
   */
  dlib::alias_tensor_instance filters() {
    return filters_(params, 0);
  }

  /**
   * @brief biases of the convolution filter
   * @return biases
   */
  dlib::alias_tensor_instance biases() {
    return biases_(params, filters_.size());
  }

  /**
   * @brief load convolution parameters from a file
   * @param filename filename
   * @return true if success to load else false
   */
  bool load_from_file(const std::string& filename) {
    std::ifstream ifs(filename);
    if(!ifs) {
      std::cerr << "error : failed to open the parameter file!!" << std::endl;
      std::cerr << "      : " << filename << std::endl;
      return false;
    }

    // read kernel
    std::string line;
    std::getline(ifs, line);

    std::vector<std::string> tokens;
    boost::split(tokens, line, boost::is_space());

    // kernel size check
    std::vector<long> filter_shape = {filters_.num_samples(), filters_.k(), filters_.nr(), filters_.nc()};
    for(int i=0; i<tokens.size(); i++) {
      if(std::stoi(tokens[i]) != filter_shape[i]) {
        std::cerr << "error : kernel_size mismatch!!" << std::endl;
        std::cerr << "      : shape[" << i << "] " << filter_shape[i] << " - " << tokens[i] << std::endl;
        return false;
      }
    }

    tokens.clear();
    std::getline(ifs, line);
    boost::split(tokens, line, boost::is_space());

    if(tokens.size() != filters_.size()) {
      std::cerr << "error : wrong number of elements!!" << std::endl;
      std::cerr << "      : " << tokens.size() << " - " << filters_.size() << std::endl;
    }

    auto f = filters();
    std::transform(tokens.begin(), tokens.end(), f.host(), [=](const std::string& v) { return std::stof(v); });

    // read biases
    tokens.clear();
    std::getline(ifs, line);
    boost::split(tokens, line, boost::is_space());

    if(tokens.empty() || std::stoi(tokens.front()) != biases_.k()) {
      std::cerr << "error : bieases_size mismatch!!" << std::endl;
      return false;
    }

    tokens.clear();
    std::getline(ifs, line);
    boost::split(tokens, line, boost::is_space());

    if(tokens.size() != biases_.size()) {
      std::cerr << "error : wrong number of elements!!" << std::endl;
      std::cerr << "      : " << tokens.size() << " - " << biases_.size() << std::endl;
    }

    auto b = biases();
    std::transform(tokens.begin(), tokens.end(), b.host(), [=](const std::string& v) { return std::stof(v); });

    return true;
  }

private:
  dlib::tensor& params;
  dlib::alias_tensor filters_;
  dlib::alias_tensor biases_;
};


/**
 * @brief The first two convolution layers of ahmed's network (tied convolution 1 and 2)
 * @see https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf
 */
template<int conv_t1_size, int conv_t2_size>
class AhmedSubnet {
public:
  using net_type =
    dlib::con<conv_t2_size, 5, 5, 1, 1,
    dlib::max_pool<2, 2, 2, 2,
    dlib::relu<
    dlib::con<conv_t1_size, 5, 5, 1, 1,
    dlib::input_rgb_image>>>>;

  /**
   * @brief constructor
   * @param directory where parameter files are contained
   */
  AhmedSubnet(const std::string& params_dir, bool use_only_first_layer = false)
    : use_only_first_layer(use_only_first_layer),
      net(dlib::input_rgb_image(0.0f, 0.0f, 0.0f))
  {
    // to determine filter sizes, run the network once
    cv::Mat dummy(160, 60, CV_8UC3);
    forward(dummy);

    // load convolution parameters
    ConvParamsProxy conv_t1(dlib::layer<3>(net).layer_details());
    ConvParamsProxy conv_t2(dlib::layer<0>(net).layer_details());

    conv_t1.load_from_file(params_dir + "/conv_t1");
    conv_t2.load_from_file(params_dir + "/conv_t2");
  }

  /**
   * @brief apply the network
   * @param rgb_image  RGB image
   * @return extracted features
   */
  std::vector<cv::Mat> forward (const cv::Mat& rgb_image) {
    dlib::cv_image<dlib::rgb_pixel> dlib_image(rgb_image);
    dlib::matrix<dlib::rgb_pixel> input;
    dlib::assign_image(input, dlib_image);

    dlib::resizable_tensor output = net(input);
    if(use_only_first_layer) {
      output = dlib::layer<2>(net).get_output();
    }

    std::vector<cv::Mat> layers(output.k());

    for(int k=0; k<output.k(); k++) {
      cv::Mat& layer = layers[k];
      layer = cv::Mat(output.nr(), output.nc(), CV_32FC1, output.host() + (k * output.nr() * output.nc())).clone();
    }

    return layers;
  }

  /**
   * @brief apply the network (identical to #forward)
   * @param rgb_image  RGB image
   * @return extracted features
   */
  std::vector<cv::Mat> operator() (const cv::Mat& rgb_image) {
    return forward(rgb_image);
  }

private:
  bool use_only_first_layer;
  net_type net;
};


/**
 * @brief Convolution features extractor
 */
template<int conv_t1_size, int conv_t2_size>
class CNNChannelExtractor : public ChannelExtractor {
public:

  /**
   * @brief constructor
   * @param params_dir  directory where the network parameters are contained
   */
  CNNChannelExtractor(const std::string& params_dir, bool use_only_first_layer = false)
    : subnet(params_dir, use_only_first_layer)
  {}
  virtual ~CNNChannelExtractor() override {}

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
   * @note since dlib's network performs in-place update during forward path, this method is not thread-safe
   */
  std::vector<cv::Mat> extract(const cv::Mat& bgr, const cv::Mat& gray) const override {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, CV_BGR2RGB);

    std::vector<cv::Mat> layers = subnet(rgb);

    for(int i=0; i<layers.size(); i++) {
      layers[i].convertTo(layers[i], CV_8UC3, 128.0);
    }

    return layers;
  }

private:
  mutable AhmedSubnet<conv_t1_size, conv_t2_size> subnet;
};

}

#endif // DCNN_CHANNEL_EXTRACTOR_HPP
