#ifndef BODY_FEATURES_HPP
#define BODY_FEATURES_HPP

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ccf_person_identification/online_classifier.hpp>


namespace ccf_person_classifier {

class BodyFeatures : public Features {
public:
    virtual ~BodyFeatures() override {}
    using Ptr = std::shared_ptr<BodyFeatures>;

    cv::Mat color;
    std::vector<cv::Mat> feature_maps;    // feature maps (i.e., feature channels)
    std::vector<cv::Mat> integral_maps;   // integral images of feature_maps};
};

}

#endif // BODY_FEATURES_HPP
