#include <iostream>
#include <ros/package.h>

#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <kkl/cvk/icf_channel_bank.hpp>
#include <ccf_person_identification/body/cnn_channel_extractor.hpp>
#include <ccf_person_identification/body/cnn_channel_extractor_gpu.hpp>


int main(int argc, char** argv) {
    std::string package_path = ros::package::getPath("ccf_person_identification");
    std::string data_dir = package_path + "/data";
    std::string dataset_dir = package_path + "/data/test";

    std::unique_ptr<cvk::ChannelBank> channel_bank(new cvk::ChannelBank());
    channel_bank->addExtractor(std::make_shared<cvk::CNNChannelExtractor<10, 10>>(data_dir + "/cnn_params_tiny"));
    // channel_bank->addExtractor(std::make_shared<cvk::CNNChannelExtractorGPU<10, 10>>(data_dir + "/cnn_params_tiny"));

    // train the classifier with the first ten frames, and test it with the rest frames
    for(int i=1; i<=14; i++) {
        cv::Mat bgr = cv::imread((boost::format("%s/p%02d.jpg") % dataset_dir % i).str());
        cv::resize(bgr, bgr, cv::Size(128, 256));

        if(!bgr.data) {
            std::cerr << "error : failed to open image!! image_id: " << i << std::endl;
            return 1;
        }

        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        // extract features
        std::vector<cv::Mat> feature_maps = channel_bank->extract(bgr, gray);

        // visualization
        for(auto& feature_map : feature_maps) {
            cv::cvtColor(feature_map, feature_map, cv::COLOR_GRAY2BGR);
            cv::resize(feature_map, feature_map, bgr.size());
        }
        feature_maps.push_back(bgr);

        cv::Mat canvas;
        cv::hconcat(feature_maps, canvas);
        cv::imshow("features", canvas);
        cv::waitKey(0);
    }
}
