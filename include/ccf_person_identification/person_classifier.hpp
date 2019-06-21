#ifndef PERSON_CLASSIFIER_HPP
#define PERSON_CLASSIFIER_HPP

#include <vector>
#include <ccf_person_identification/online_classifier.hpp>
#include <ccf_person_identification/face/face_classifier.hpp>
#include <ccf_person_identification/body/body_classifier.hpp>


namespace ccf_person_classifier {

class PersonInput : public BodyInput, public FaceInput {
public:
    virtual ~PersonInput() override {}
    using Ptr = std::shared_ptr<PersonInput>;
};

class PersonFeatures : public BodyFeatures, public FaceFeatures {
public:
    virtual ~PersonFeatures() override {}
    using Ptr = std::shared_ptr<PersonFeatures>;
};


class PersonClassifier : public OnlineClassifier {
public:
    PersonClassifier(ros::NodeHandle& nh)
    {
        if(nh.param<bool>("use_face", true)) {
            classifiers.push_back(std::make_shared<FaceClassifier>(nh));
        }
        if(nh.param<bool>("use_body", true)) {
            classifiers.push_back(std::make_shared<BodyClassifier>(nh));
        }
    }

    virtual ~PersonClassifier() override {}

    virtual std::string name() const override {
        return "face+body";
    }

    std::vector<std::string> classifierNames() const {
        std::vector<std::string> names;
        for(const auto& classifier: classifiers) {
            names.push_back(classifier->name());
        }

        return names;
    }

    template<typename Classifier>
    std::shared_ptr<Classifier> getClassifier(const std::string& name) const {
        for(const auto& classifier : classifiers) {
            if(name == classifier->name()) {
                return std::dynamic_pointer_cast<Classifier>(classifier);
            }
        }

        return nullptr;
    }

    virtual bool extractInput(Input::Ptr& input, const std::unordered_map<std::string, cv::Mat>& images) override {
        std::vector<bool> extracted(classifiers.size(), false);
        std::transform(classifiers.begin(), classifiers.end(), extracted.begin(),
            [&](const OnlineClassifier::Ptr& classifier) {
                return classifier->extractInput(input, images);
            }
        );

        return std::any_of(extracted.begin(), extracted.end(), [=](bool b) { return b; });
    }

    virtual bool extractFeatures(Features::Ptr& features, const Input::Ptr& input) override {
        std::vector<bool> extracted(classifiers.size(), false);
        std::transform(classifiers.begin(), classifiers.end(), extracted.begin(),
            [&](const OnlineClassifier::Ptr& classifier) {
                return classifier->extractFeatures(features, input);
            }
        );

        return std::any_of(extracted.begin(), extracted.end(), [=](bool b) { return b; });
    }

    virtual bool update(double label, const Features::Ptr& features) override {
        std::vector<bool> updated(classifiers.size(), false);
        std::transform(classifiers.begin(), classifiers.end(), updated.begin(),
            [&](const OnlineClassifier::Ptr& classifier) {
                return classifier->update(label, features);
            }
        );

        return std::any_of(updated.begin(), updated.end(), [=](bool b) { return b; });
    }

    virtual boost::optional<double> predict(const Features::Ptr& features) override {
        ROS_ERROR_STREAM("this method must not be called!!");
        abort();
        return boost::none;
    }

    boost::optional<double> predict(const Features::Ptr& features, std::vector<double>& classifier_confidences) {
        std::vector<boost::optional<double>> results(classifiers.size(), false);
        std::transform(classifiers.begin(), classifiers.end(), results.begin(),
            [&](const OnlineClassifier::Ptr& classifier) {
                return classifier->predict(features);
            }
        );

        if(classifier_confidences.empty()) {
            classifier_confidences.resize(results.size(), -0.2);
        }
        for(int i=0; i<results.size(); i++) {
            if(results[i]) {
                classifier_confidences[i] = *results[i];
            }
        }

        boost::optional<double> aggregated = results[0];

        if(aggregated) {
            for(int i=1; i<results.size(); i++) {
                double conf = results[i] ? *results[i] : classifier_confidences[i];
                *aggregated += conf;
            }

            *aggregated /= results.size();
        }

        return aggregated;
    }

private:
    std::vector<OnlineClassifier::Ptr> classifiers;
};

}

#endif // PERSON_CLASSIFIER_HPP
