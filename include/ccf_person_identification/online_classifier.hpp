#ifndef ONLINE_CLASSIFIER_HPP
#define ONLINE_CLASSIFIER_HPP

#include <memory>
#include <unordered_map>
#include <boost/optional.hpp>
#include <opencv2/opencv.hpp>


namespace ccf_person_classifier {

class Input {
public:
    virtual ~Input() {}

    using Ptr = std::shared_ptr<Input>;
};

class Features {
public:
    virtual ~Features() {}

    using Ptr = std::shared_ptr<Features>;
};


class OnlineClassifier {
public:
    virtual ~OnlineClassifier() {}

    using Ptr = std::shared_ptr<OnlineClassifier>;

    /**
     * @brief classifier name
     * @return
     */
    virtual std::string name() const = 0;

    /**
     * @brief update classifier
     * @param features  person features
     * @return if the input is used to update the classifier
     */
    virtual bool update(double label, const Features::Ptr& features) = 0;

    /**
     * @brief classify input features
     * @param features  person features
     * @return boost::none if the classifier is not ready
     *         otherwise classification result (sign: class + or -, magnitude: confidence)
     */
    virtual boost::optional<double> predict(const Features::Ptr& features) = 0;


    /**
     * @brief extract input data from raw image
     * @param input
     * @param bgr_image
     * @return
     */
    virtual bool extractInput(Input::Ptr& input, const std::unordered_map<std::string, cv::Mat>& images) = 0;

    /**
     * @brief extract features from input data
     * @param features  in/out features
     * @return if feature is extracted
     */
    virtual bool extractFeatures(Features::Ptr& features, const Input::Ptr& input) = 0;
};


}

#endif // ONLINE_CLASSIFIER_HPP
