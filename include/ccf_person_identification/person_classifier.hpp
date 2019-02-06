#ifndef PERSON_CLASSIFIER_HPP
#define PERSON_CLASSIFIER_HPP

#include <ccf_person_identification/online_classifier.hpp>
#include <ccf_person_identification/body/body_classifier.hpp>


namespace ccf_person_classifier {

class PersonFeatures : public BodyFeatures {
    virtual ~PersonFeatures() override {}
};


class PersonClassifier : public OnlineClassifier {
public:
    PersonClassifier(ros::NodeHandle& nh)
        : body_classifier(new BodyClassifier(nh))
    {
    }

    virtual ~PersonClassifier() override {}

    virtual bool extractFeatures(Features::Ptr& features, const cv::Mat& bgr_image, const cv::Mat& gray_image) override {
        bool body_extracted = body_classifier->extractFeatures(features, bgr_image, gray_image);

        return body_extracted;
    }

    virtual bool update(double label, const Features::Ptr& features) override {
        bool body_updated = body_classifier->update(label, features);
        return body_updated;
    }

    virtual boost::optional<double> predict(const Features::Ptr& features) override {
        boost::optional<double> body_result = body_classifier->predict(features);

        return body_result;
    }

private:
    std::unique_ptr<BodyClassifier> body_classifier;
};

}

#endif // PERSON_CLASSIFIER_HPP
