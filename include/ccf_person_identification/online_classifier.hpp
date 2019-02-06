#ifndef ONLINE_CLASSIFIER_HPP
#define ONLINE_CLASSIFIER_HPP

#include <memory>
#include <boost/optional.hpp>
#include <opencv2/opencv.hpp>


namespace ccf_person_classifier {

class Features {
public:
    virtual ~Features() {}

    using Ptr = std::shared_ptr<Features>;
};


class OnlineClassifier {
public:
    virtual ~OnlineClassifier() {}

    /**
     * @brief update the classifier
     * @param features  person features
     * @return if the input is used to update the classifier
     */
    virtual bool update(double label, const Features::Ptr& features) = 0;

    /**
     * @brief classify the input features
     * @param features  person features
     * @return boost::none if the classifier is not ready
     *         otherwise classification result (sign: class + or -, magnitude: confidence)
     */
    virtual boost::optional<double> predict(const Features::Ptr& features) = 0;


    /**
     * @brief extractFeatures
     * @param features  in/out features
     * @return if feature is extracted
     */
    virtual bool extractFeatures(Features::Ptr& features, const cv::Mat& bgr_image, const cv::Mat& gray_image) = 0;
};


}

#endif // ONLINE_CLASSIFIER_HPP
