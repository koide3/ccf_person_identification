#include <ccf_person_identification/face/face_features.hpp>

#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <kkl/ml/flann_knn_classifier.hpp>
#include <open_face_recognition/CalcImageFeatures.h>


namespace ccf_person_classifier {

class FaceClassifier : public virtual OnlineClassifier {
public:
    using Ptr = std::shared_ptr<FaceClassifier>;

    FaceClassifier(ros::NodeHandle& nh)
        : face_detector(dlib::get_frontal_face_detector()),
          client(nh.serviceClient<open_face_recognition::CalcImageFeatures>("/calc_image_features"))
    {
        knn_k = 5;
        num_positives = 0;
        num_negatives = 0;
        classifier.reset(new kkl::ml::FlannKNNClassifier<float>());
    }

    virtual ~FaceClassifier() override {}

    virtual std::string name() const override {
        return "face";
    }

    virtual bool update(double label, const Features::Ptr& features_) {
        FaceFeatures::Ptr features = std::dynamic_pointer_cast<FaceFeatures>(features_);
        if(!features) {
            return false;
        }

        float min_dist = 0.0f;
        double pred = classifier->predictBinaryReal(features->face_features, knn_k, &min_dist);

        if(label > 0.0) {
            num_positives ++;
        } else {
            num_negatives ++;
        }

        if((pred > 0.0) != (label > 0.0) || std::abs(pred) < 0.6 || min_dist > 0.2) {
            classifier->addPoint(label, features->face_features);
        }
    }

    virtual boost::optional<double> predict(const Features::Ptr& features_) override {
        FaceFeatures::Ptr features = std::dynamic_pointer_cast<FaceFeatures>(features_);
        if(!features) {
            return boost::none;
        }

        return classifier->predictBinaryReal(features->face_features, knn_k);
    }

    virtual bool extractInput(Input::Ptr& input_, const cv::Mat& bgr_image) override {
        FaceInput::Ptr input = std::dynamic_pointer_cast<FaceInput>(input_);
        if(!input) {
            return false;
        }

        double face_roi_height = 0.33;
        double face_roi_width = 0.8;
        input->face_roi = cv::Rect(bgr_image.cols * (1 - face_roi_width) / 2, 0, bgr_image.cols * face_roi_width, bgr_image.rows * face_roi_height);

        cv::Mat face_roi_gray;
        cv::cvtColor(cv::Mat(bgr_image, input->face_roi), face_roi_gray, cv::COLOR_BGR2GRAY);

        dlib::cv_image<uchar> dgray(face_roi_gray);
        std::vector<dlib::rect_detection> faces;
        face_detector(dgray, faces);

        if(faces.empty()) {
            return false;
        }

        auto largest = std::max_element(faces.begin(), faces.end(), [=](const dlib::rect_detection& lhs, const dlib::rect_detection& rhs) { return lhs.rect.area() < rhs.rect.area(); });
        input->face_region = cv::Rect(largest->rect.left(), largest->rect.top(), largest->rect.width(), largest->rect.height());
        input->face_image = cv::Mat(bgr_image, *input->face_region);

        return true;
    }

    virtual bool extractFeatures(Features::Ptr& features_, const Input::Ptr& input_) override {
        FaceInput::Ptr input = std::dynamic_pointer_cast<FaceInput>(input_);
        FaceFeatures::Ptr features = std::dynamic_pointer_cast<FaceFeatures>(features_);
        if(!input || !features) {
            return false;
        }

        open_face_recognition::CalcImageFeatures srv;
        cv_bridge::CvImage cv_image(std_msgs::Header(), "bgr8", input->face_image);
        cv_image.toImageMsg(srv.request.image);

        if(!client.call(srv)) {
          std::cerr << "error : failed to extract face features!!" << std::endl;
          std::cerr << "      : open_face_recognition_node may be stopping!!" << std::endl;
          return false;
        }

        if(srv.response.features.data.size() < 2) {
          return false;
        }

        features->face_features.resize(srv.response.features.data.size());
        memcpy(features->face_features.data(), srv.response.features.data.data(), sizeof(float) * features->face_features.size());

        return true;
    }

private:
    dlib::frontal_face_detector face_detector;

    int knn_k;
    int num_positives;
    int num_negatives;
    std::unique_ptr<kkl::ml::FlannKNNClassifier<float>> classifier;

    ros::ServiceClient client;
};


}
