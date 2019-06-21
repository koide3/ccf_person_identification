/**
 * FlannKNNClassifier.hpp
 * @author koide
 * 15/11/16
 **/
#ifndef KKL_FLANN_KNN_CLASSIFIER_HPP
#define KKL_FLANN_KNN_CLASSIFIER_HPP

#undef USE_UNORDERED_MAP

#include <deque>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <flann/flann.hpp>

namespace kkl {
namespace ml {

/**
 * @brief k-nearest neighbor classifier based on Flann
 */
template<typename T>
class FlannKNNClassifier {
public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorT;

    /**
     * @brief constructor
     */
    FlannKNNClassifier() {
        index_params.reset(new flann::LinearIndexParams());

        labels.reserve(256);

        min_label = 0;
        max_label = 0;
    }

    /**
     * @brief if the classifier is ready
     * @return if the classifier is ready
     */
    bool ready() const {
        return index != nullptr;
    }

    /**
     * @brief add a point to the classifier
     * @param label  label
     * @param point  feature vector
     */
    void addPoint(int label, const VectorT& point){
        labels.push_back(label);
        points.push_back(point);

        if (!index) {
            index.reset(new ::flann::Index<::flann::L2<float>>(eigen2flann(points.back()), *index_params));
        } else {
            index->addPoints(eigen2flann(points.back()), 1);
        }

        min_label = std::min(min_label, label);
        max_label = std::max(max_label, label);
    }

    /**
     * @brief classify a point into trained classes
     * @param query   query feature vector
     * @param k       number of neighbors to be used for classification
     * @return class
     */
    int predict(const VectorT& query, int k = 5) {
        if (!index) {
            std::cerr << "error : knn index is not constructed!!" << std::endl;
            return min_label;
        }

        std::vector<std::vector<int>> indices_;
        std::vector<std::vector<float>> dists_;
        index->knnSearch(eigen2flann(query), indices_, dists_, k, flann::SearchParams(32));

        const auto& indices = indices_.front();
        const auto& dists = dists_.front();

        int label_range = max_label - min_label + 1;
        if (label_range == 1) {
            return min_label;
        }

        std::vector<int> hist(label_range, 0);
        for (int i = 0; i < indices.size(); i++) {
            hist[labels[indices[i]] + min_label] ++;
        }

        int max_voted = min_label + std::distance(hist.begin(), std::max_element(hist.begin(), hist.end()));
        return max_voted;
    }

    /**
     * @brief binary classification
     * @param query     query feature vector
     * @param k         number of neighbors to be used for classification
     * @param min_dist  [out] distance to the closest point
     * @return label > 1 or not
     */
    bool predictBinary(const VectorT& query, int k = 5, float* min_dist = nullptr) {
        if (!index) {
            std::cerr << "error : knn index is not constructed!!" << std::endl;
            return min_label;
        }

        std::vector<std::vector<int>> indices_;
        std::vector<std::vector<float>> dists_;
        index->knnSearch(eigen2flann(query), indices_, dists_, k, flann::SearchParams(32));

        const auto& indices = indices_.front();
        const auto& dists = dists_.front();

        int pos_neg[2] = { 0, 0 };
        for (int i = 0; i < indices.size(); i++) {
            if (labels[indices[i]] > 0) {
                pos_neg[0]++;
            }
            else {
                pos_neg[1] ++;
            }
        }

        if (min_dist) {
            *min_dist = dists[0];
            std::cout << "min_dist " << *min_dist << std::endl;
        }

        return pos_neg[0] > pos_neg[1];;
    }

    /**
     * @brief classification with confidence
     * @param query      query feature vector
     * @param k          num. neighbors
     * @param min_dist   distance to the closest point
     * @return sign: class, magnitude: confidence [-1, 1]
     */
    double predictBinaryReal(const VectorT& query, int k = 5, float* min_dist = nullptr) {
        if (!index) {
            std::cerr << "error : knn index is not constructed!!" << std::endl;
            return min_label;
        }

        std::vector<std::vector<int>> indices_;
        std::vector<std::vector<float>> dists_;
        index->knnSearch(eigen2flann(query), indices_, dists_, k, flann::SearchParams(32));

        const auto& indices = indices_.front();
        const auto& dists = dists_.front();

        int pos_neg[2] = { 0, 0 };
        for (int i = 0; i < indices.size(); i++) {
            if (labels[indices[i]] > 0) {
                pos_neg[0]++;
            }
            else {
                pos_neg[1] ++;
            }
        }

        if (min_dist) {
            *min_dist = dists[0];
        }

        double sign = pos_neg[0] > pos_neg[1] ? +1.0 : -1.0;
        int half = (k - 1) / 2;
        int range = k - half;
        double confidence = (std::max(pos_neg[0], pos_neg[1]) - half) / static_cast<double>(range);

        return sign * confidence;
    }

    /**
     * @brief number of points in index
     * @return
     */
    size_t size() const { return labels.size(); }

private:
    // Eigen::Matrix -> flann::Matrix
    flann::Matrix<T> eigen2flann(const VectorT& v) {
        flann::Matrix<T> mat((T*)v.data(), 1, v.size());
        return mat;
    }

    int min_label;				// minimum label
    int max_label;				// maximum label
    std::vector<int> labels;	// label set
    std::deque<VectorT> points;	// point set

    // flann
    std::unique_ptr<flann::IndexParams> index_params;
    std::unique_ptr<flann::Index<flann::L2<T>>> index;
};

}
}

#endif
