#ifndef KKL_ICF_INTEGRAL_FILTER_HPP
#define KKL_ICF_INTEGRAL_FILTER_HPP

#include <opencv2/opencv.hpp>

namespace cvk {

/**
 * @brief A class to calculate sum of pixel values in an ROI using integral image
 */
class IntegralFilter {
public:
  /**
   * @brief constructor
   * @param tl    top left position of the ROI
   * @param size  size of the ROI
   */
  IntegralFilter(const cv::Point2f& tl, const cv::Size2f& size)
    : rect_tl(tl),
      rect_size(size)
  {
    const double minimum_size = 0.05;
        assert(tl.x >= 0.0 && tl.x <= 1.0 - minimum_size);
        assert(tl.y >= 0.0 && tl.y <= 1.0 - minimum_size);
        assert(size.width >= minimum_size && (tl.x + size.width) <= 1.0);
        assert(size.height >= minimum_size && (tl.y + size.height) <= 1.0);
  }

  virtual ~IntegralFilter() {}

  /**
   * @brief calculate the sum of pixel values in the ROI
   * @param integral  integral image of the target image
   * @return the sum of pixel values in the ROI
   */
  double filter(const cv::Mat& integral) const {
    cv::Rect rect = calcRect(integral.size());
    if(rect.width <= 2 || rect.height <= 2) {
      return 0.0;
    }

    int a = rect.y <= 0 || rect.x <= 0 ? 0 : integral.at<int>(rect.y-1, rect.x-1);
    int c = rect.y <= 0 ? 0 : integral.at<int>(rect.y-1, rect.x-1 + rect.width);
    int b = rect.x <= 0 ? 0 : integral.at<int>(rect.y-1 + rect.height, rect.x-1);
    int d = integral.at<int>(rect.y-1 + rect.height, rect.x-1 + rect.width);

    return static_cast<double>(d - b - c + a) / rect.area();

  }

  cv::Rect calcRect(const cv::Size& size) const {
    return cv::Rect(rect_tl.x * size.width, rect_tl.y * size.height, rect_size.width * size.width, rect_size.height * size.height);
  }

  const cv::Point2f& tl() const { return rect_tl; }
  const cv::Size2f& size() const { return rect_size; }

private:
  cv::Point2f rect_tl;
  cv::Size2f rect_size;
};

}

#endif // INTEGRAL_FILTER_HPP
