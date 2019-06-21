#ifndef FACE_FEATURES_HPP
#define FACE_FEATURES_HPP

#include <Eigen/Dense>
#include <ccf_person_identification/online_classifier.hpp>

namespace ccf_person_classifier {

class FaceInput : public virtual Input {
public:
    virtual ~FaceInput() {}
    using Ptr = std::shared_ptr<FaceInput>;

    cv::Mat face_image;
};


class FaceFeatures : public virtual Features {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual ~FaceFeatures() override {}
    using Ptr = std::shared_ptr<FaceFeatures>;

    Eigen::VectorXf face_features;
};

}

#endif // FACE_FEATURES_HPP
