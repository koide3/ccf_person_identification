#include <iostream>
#include <ros/package.h>

#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <ccf_person_identification/body/body_classifier.hpp>


using namespace ccf_person_classifier;


int main(int argc, char** argv) {
    ros::init(argc, argv, "ccf_person_identification_test");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    std::unique_ptr<BodyClassifier> classifier(new BodyClassifier(private_nh));

    std::string package_path = ros::package::getPath("ccf_person_identification");
    std::string dataset_dir = package_path + "/data/test";

    // train the classifier with the first ten frames, and test it with the rest frames
    for(int i=1; i<=14; i++) {
        cv::Mat pos = cv::imread((boost::format("%s/p%02d.jpg") % dataset_dir % i).str());
        cv::Mat neg1 = cv::imread((boost::format("%s/n%02d-01.jpg") % dataset_dir % i).str());
        cv::Mat neg2 = cv::imread((boost::format("%s/n%02d-02.jpg") % dataset_dir % i).str());

        if(!pos.data || !neg1.data || !neg2.data) {
            std::cerr << "error : failed to open image!! image_id: " << i << std::endl;
            return 1;
        }

        cv::Mat pos_gray;
        cv::cvtColor(pos, pos_gray, cv::COLOR_BGR2GRAY);

        cv::Mat neg1_gray;
        cv::cvtColor(neg1, neg1_gray, cv::COLOR_BGR2GRAY);

        cv::Mat neg2_gray;
        cv::cvtColor(neg2, neg2_gray, cv::COLOR_BGR2GRAY);


        cv::Mat pos_result;
        cv::resize(pos, pos_result, cv::Size(128, 256));

        cv::Mat neg1_result;
        cv::resize(neg1, neg1_result, cv::Size(128, 256));

        cv::Mat neg2_result = neg2.clone();
        cv::resize(neg2, neg2_result, cv::Size(128, 256));

        Features::Ptr features(new BodyFeatures());
        classifier->extractFeatures(features, pos, pos_gray);
        cv::Scalar pos_color = *classifier->predict(features) > 0.0 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
        cv::rectangle(pos_result, cv::Point(0, 0), cv::Point(128, 256), pos_color, 5);

        ROS_INFO_STREAM("pos:" << *classifier->predict(features));

        classifier->extractFeatures(features, neg1, neg1_gray);
        cv::Scalar neg1_color = *classifier->predict(features) > 0.0 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
        cv::rectangle(neg1_result, cv::Point(0, 0), cv::Point(128, 256), neg1_color, 5);

        classifier->extractFeatures(features, neg2, neg2_gray);
        cv::Scalar neg2_color = *classifier->predict(features) > 0.0 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
        cv::rectangle(neg2_result, cv::Point(0, 0), cv::Point(128, 256), neg2_color, 5);

        std::vector<cv::Mat> results = {pos_result, neg1_result, neg2_result};
        cv::Mat canvas;
        cv::hconcat(results, canvas);
        cv::putText(canvas, i <= 10 ? "training" : "testing", cv::Point(10, 25), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar::all(0), 2);
        cv::putText(canvas, i <= 10 ? "training" : "testing", cv::Point(10, 25), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar::all(255));
        cv::imshow("results", canvas);

        cv::Mat feature_map = classifier->visualize();
        if(feature_map.data) {
            cv::imshow("feature_map", feature_map);
        }
        cv::waitKey(0);

        if(i <= 10)  {
            classifier->extractFeatures(features, pos, pos_gray);
            classifier->update(1.0, features);

            classifier->extractFeatures(features, neg1, neg1_gray);
            classifier->update(-1.0, features);

            classifier->extractFeatures(features, neg2, neg2_gray);
            classifier->update(-1.0, features);
        }
    }

    return 0;
}
