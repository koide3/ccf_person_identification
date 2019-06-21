#ifndef BODY_CLASSIFIER_HPP
#define BODY_CLASSIFIER_HPP

#include <boost/algorithm/string.hpp>

#include <ros/ros.h>
#include <ros/package.h>

#include <kkl/ml/online_boosting.hpp>
#include <kkl/cvk/icf_channel_bank.hpp>

#include <ccf_person_identification/online_classifier.hpp>
#include <ccf_person_identification/body/body_features.hpp>
#include <ccf_person_identification/body/body_weak_classifier.hpp>
#include <ccf_person_identification/body/cnn_channel_extractor.hpp>


namespace ccf_person_classifier {


class BodyClassifier : public virtual OnlineClassifier {
public:
    using Ptr = std::shared_ptr<BodyClassifier>;

    BodyClassifier(ros::NodeHandle& nh)
        : num_positives(0),
          num_negatives(0)
    {
        std::string package_path = ros::package::getPath("ccf_person_identification");

        std::string channel_types = nh.param<std::string>("channel_types", "cnn10");
        bool use_only_first_layer = nh.param<bool>("use_only_first_layer", false);

        // define channel set for body classification
        ROS_INFO("construct body classifier");
        channel_bank.reset(new cvk::ChannelBank());
        if(channel_types.find("luv") != std::string::npos) {
          ROS_INFO("add LUV to channel bank");
          channel_bank->addExtractor(std::make_shared<cvk::ChannelExtractorLUV>());
        }
        if(channel_types.find("grads") != std::string::npos) {
          ROS_INFO("add grads to channel bank");
          channel_bank->addExtractor(std::make_shared<cvk::ChannelExtractorGrads>());
        }
        if(channel_types.find("cnn10") != std::string::npos) {
          ROS_INFO("add cnn10 to channel bank");
          channel_bank->addExtractor(std::make_shared<cvk::CNNChannelExtractor<10, 10>>(package_path + "/data/cnn_params_tiny", use_only_first_layer));
        }
        if(channel_types.find("cnn25") != std::string::npos) {
          ROS_INFO("add cnn25 to channel bank");
          channel_bank->addExtractor(std::make_shared<cvk::CNNChannelExtractor<20, 25>>(package_path + "/data/cnn_params", use_only_first_layer));
        }

        // create an online boosting classifier
        generator.reset(new BodyWeakClassifierGenerator(channel_bank->numChannels(), 0.2, 0.5));

        std::vector<double> init_labels;
        std::vector<BodyFeatures::Ptr> init_samples;
        int num_selectors = nh.param<int>("num_selectors", 15);
        int num_weak_classifiers = nh.param<int>("num_weak_classifiers", 15);
        boosting.reset(new kkl::ml::OnlineBoosting<BodyFeatures::Ptr>(generator, num_selectors, num_weak_classifiers, init_labels, init_samples, 2, 32));

    }
    virtual ~BodyClassifier() override {}


    virtual std::string name() const override {
        return "body";
    }

    virtual bool extractInput(Input::Ptr& input_, const std::unordered_map<std::string, cv::Mat>& images) override {
        auto found = images.find("body");
        if(found == images.end()) {
            return false;
        }
        const auto& bgr_image = found->second;

        BodyInput::Ptr input = std::dynamic_pointer_cast<BodyInput>(input_);
        if(!input) {
            return false;
        }

        input->bgr_image = bgr_image.clone();
        cv::cvtColor(bgr_image, input->gray_image, cv::COLOR_BGR2GRAY);

        return true;
    }

    virtual bool extractFeatures(Features::Ptr& features_, const Input::Ptr& input_) override {
        BodyInput::Ptr input = std::dynamic_pointer_cast<BodyInput>(input_);
        BodyFeatures::Ptr features = std::dynamic_pointer_cast<BodyFeatures>(features_);
        if(!input || !features) {
            return false;
        }

        features->color = input->bgr_image;
        features->feature_maps = channel_bank->extract(input->bgr_image, input->gray_image);
        features->integral_maps.resize(features->feature_maps.size());
        for(size_t i=0; i<features->feature_maps.size(); i++) {
          cv::integral(features->feature_maps[i], features->integral_maps[i]);
        }

        return true;
    }


    virtual bool update(double label, const Features::Ptr& features_) override {
        BodyFeatures::Ptr features = std::dynamic_pointer_cast<BodyFeatures>(features_);
        if(!features) {
            return false;
        }

        if(label > 0.0) {
          num_positives ++;
          last_target = features;
        } else {
          num_negatives ++;
        }
        boosting->update(label, features);

        return false;
    }

    virtual boost::optional<double> predict(const Features::Ptr& features_) override {
        BodyFeatures::Ptr features = std::dynamic_pointer_cast<BodyFeatures>(features_);
        if(!features) {
            return false;
        }

        return boosting->predictReal(features);
    }


    cv::Mat visualize() const {
        if(!last_target) {
            return cv::Mat();
        }

        int num_channels = channel_bank->numChannels();
        cv::Size map_size(64, 128);

        std::vector<cv::Mat> feature_maps(last_target->feature_maps.size() + 1);
        for(int i=0; i<feature_maps.size(); i++) {
            if(i < last_target->feature_maps.size()) {
                cv::resize(last_target->feature_maps[i], feature_maps[i], map_size);
            } else {
                cv::resize(last_target->color, feature_maps[i], map_size);
            }
        }

        std::vector<cv::Mat> response_maps(num_channels + 1);
        for(auto& map : response_maps) {
          map = cv::Mat(map_size.height, map_size.width, CV_8UC1, cv::Scalar::all(0));
        }

        int layer_intensity = 500 / num_channels;
        cv::Mat layer(map_size.height, map_size.width, CV_8UC1);

        const auto& selectors = boosting->selectors;

        for(int i=0; i<selectors.size(); i++) {
          std::shared_ptr<BodyWeakClassifier> weak_classifier = std::dynamic_pointer_cast<BodyWeakClassifier>(selectors[i]->bestClassifier());
          if(!weak_classifier) {
              continue;
          }
          int channel_num = weak_classifier->channelNum();
          cv::Rect rect = weak_classifier->calcRect(response_maps[channel_num].size());

          layer = cv::Scalar::all(0);
          cv::rectangle(layer, rect, cv::Scalar::all(layer_intensity), -1);

          response_maps[channel_num] += layer;
          response_maps[num_channels] += layer;
        }

        auto channel_names = channel_bank->channelNames();
        channel_names.push_back("all");

        cv::Mat canvas(map_size.height, map_size.width * (num_channels + 1), CV_8UC3);
        for(int i=0; i<num_channels + 1; i++){
          cv::Mat roi(canvas, cv::Rect(i * map_size.width, 0, map_size.width, map_size.height));
          cv::Mat jet;
          cv::applyColorMap(response_maps[i], jet, cv::COLORMAP_JET);

          cv::Mat feat;
          if(i != num_channels) {
            cv::cvtColor(feature_maps[i], feat, CV_GRAY2BGR);
          } else {
            feat = feature_maps.back().clone();
          }
          cv::resize(feat, feat, cv::Size(map_size.width, map_size.height));

          cv::addWeighted(jet, 0.5, feat, 0.5, 0.0, roi);
          cv::putText(roi, channel_names[i], cv::Point(4, 10), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar::all(255));
        }

        return canvas;
    }

private:
    int num_positives;
    int num_negatives;
    BodyFeatures::Ptr last_target;

    std::unique_ptr<cvk::ChannelBank> channel_bank;
    std::shared_ptr<BodyWeakClassifierGenerator> generator;

    std::unique_ptr<kkl::ml::OnlineBoosting<BodyFeatures::Ptr>> boosting;
};


}

#endif
