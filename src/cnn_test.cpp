#include <iostream>
#include <ros/package.h>

#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <kkl/cvk/icf_channel_bank.hpp>
#include <ccf_person_identification/body/cnn_channel_extractor.hpp>
#include <ccf_person_identification/body/cnn_channel_extractor_gpu.hpp>


int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "usage: cnn_test input_filename (output_filename)" << std::endl;
        return 0;
    }
    std::string input_filename = argv[1];
    cv::Mat image = cv::imread(input_filename);

    if(!image.data) {
        std::cerr << "failed to read the image" << std::endl;
        return 1;
    }

    std::string output_filename = "/tmp/featuers.png";
    if(argc >= 3) {
        output_filename = argv[2];
    }

    std::string package_path = ros::package::getPath("ccf_person_identification");
    std::string data_dir = package_path + "/data";
    std::string dataset_dir = package_path + "/data/test";

    std::unique_ptr<cvk::ChannelBank> channel_bank(new cvk::ChannelBank());
    channel_bank->addExtractor(std::make_shared<cvk::CNNChannelExtractor<10, 10>>(data_dir + "/cnn_params_tiny"));

    cv::Mat bgr = image.clone();
    cv::resize(bgr, bgr, cv::Size(128, 256));

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
    cv::imwrite(output_filename, canvas);
    cv::waitKey(0);
}
