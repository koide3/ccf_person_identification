/** 
 * OnlineBoosting.hpp
 * @author koide
 * 15/01/08
 **/
#ifndef KKL_ONLINE_BOOSTING_HPP
#define KKL_ONLINE_BOOSTING_HPP

#include <cmath>
#include <random>
#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>
#include <boost/circular_buffer.hpp>

namespace kkl{
namespace ml{

/*******************************************
 * WeakClassifier
 *
 *******************************************/
template<typename Input_>
class WeakClassifier {
protected:
    typedef Input_ Input;
public:
    // constructor, destructor
    WeakClassifier() {}
    virtual ~WeakClassifier() {}

    // update classifier
    // label must not be zero
    virtual void update(double label, const Input& x) = 0;

    // predict response
    virtual double predict(const Input& x) const = 0;

    // text
    virtual std::string toString() const { return "WeakClassifier"; }
};

/*******************************************
 * WeakClassifierGenerator
 *
 *******************************************/
template<typename Input_>
class WeakClassifierGenerator {
protected:
    typedef Input_ Input;
    typedef std::shared_ptr<WeakClassifier<Input>> WeakClassifierPtr;
public:
    // constructor, destructor
    WeakClassifierGenerator() {}
    virtual ~WeakClassifierGenerator() {}

    virtual WeakClassifierPtr generate(const std::vector<double>& labels, const std::vector<Input>& samples) = 0;
};

/*******************************************
 * WeakClassifierSelector
 *
 *******************************************/
template<typename Input>
class WeakClassifierSelector {
    typedef std::shared_ptr<WeakClassifier<Input>> WeakClassifierPtr;
public:
    // constructor
    WeakClassifierSelector(int classifier_num) {
        errors.reserve(classifier_num);
        weak_classifiers.reserve(classifier_num);
        lambda_corr_wrong.reserve(classifier_num);
        voting_weight = 0.0;
        best_classifier_index = 0;
        worst_classifier_index = -1;
    }

    // add new weak classifier
    void push(WeakClassifierPtr& weak_classifier) {
        assert(weak_classifiers.size() != weak_classifiers.capacity());
        errors.push_back(0.5);
        weak_classifiers.push_back(weak_classifier);
        lambda_corr_wrong.push_back(std::make_pair(1.0, 1.0));
    }

    // replace the worst classifier with new classifier
    void replace(WeakClassifierPtr& weak_classifier) {
        if (worst_classifier_index < 0){
            return;
        }
        errors[worst_classifier_index] = 0.5;
        weak_classifiers[worst_classifier_index] = weak_classifier;
        lambda_corr_wrong[worst_classifier_index] = std::make_pair(1.0, 1.0);
    }

    // update selector
    // label = weight * label
    // x : input data
    double update(double label, const Input& x) {
        int best_index = -1;
        int worst_index = -1;
        double best_error_rate = 1.0;
        double worst_error_rate = 0.0;
        bool best_succeeded = false;

        double lambda = std::abs(label);

        // for all classifiers
        for (int i = 0; i < weak_classifiers.size(); i++) {
            // update classifier
            weak_classifiers[i]->update(label, x);

            // estimate errors
            bool success = (weak_classifiers[i]->predict(x) > 0.0 ? 1 : -1) == (label > 0.0 ? 1 : -1);
            if (success) {
                lambda_corr_wrong[i].first += lambda;
            }
            else {
                lambda_corr_wrong[i].second += lambda;
            }
            errors[i] = lambda_corr_wrong[i].second / (lambda_corr_wrong[i].first + lambda_corr_wrong[i].second);

            // find best/worst classifier
            if (errors[i] < best_error_rate) {
                best_error_rate = errors[i];
                best_index = i;
                best_succeeded = success;
            }
            if (errors[i] >= worst_error_rate) {
                worst_error_rate = errors[i];
                worst_index = i;
            }
        }

        worst_classifier_index = worst_index;
        best_classifier_index = -1;
        if (best_error_rate > 0.5 || best_error_rate == 0) {
            std::cout << "bad selector" << std::endl;
            return -1.0;
        }

        best_classifier_index = best_index;
        worst_classifier_index = worst_index;
        voting_weight = 0.5 * std::log((1 - best_error_rate) / best_error_rate);

        // return next lambda
        return best_succeeded ? lambda / (2 * (1 - best_error_rate)) : lambda / (2 * best_error_rate);
    }

    // predict response using best classifier
    double predict(const Input& x) const {
        if (best_classifier_index < 0) {
            return 0.0;
        }
        return voting_weight * weak_classifiers[best_classifier_index]->predict(x);
        //		return voting_weight * weak_classifiers[best_classifier_index]->predict(x) > 0.0 ? 1.0 : -1.0;
    }

    // return classifier which has best error rate
    const WeakClassifierPtr bestClassifier() const {
        if (best_classifier_index < 0) {
            return WeakClassifierPtr();
        }
        return weak_classifiers[best_classifier_index];
    }

    double votingWeight() const { return voting_weight; }

private:
    std::vector<double> errors;
    std::vector<WeakClassifierPtr> weak_classifiers;
    std::vector<std::pair<double, double>> lambda_corr_wrong;

    double voting_weight;
    int best_classifier_index;
    int worst_classifier_index;
};


/*******************************************
 * OnlineBoosting
 *
 *******************************************/
template<typename Input>
class OnlineBoosting {
private:
    typedef WeakClassifierSelector<Input> Selector;
    typedef WeakClassifierGenerator<Input> Generator;
public:

    OnlineBoosting(std::shared_ptr<Generator> generator, int selector_num, int weak_classifier_num, const std::vector<double>& init_labels, const std::vector<Input>& init_samples, int replace_cycle = 4, int sample_keep_num = 32)
        : generator(generator),
          latest_pos_labels(sample_keep_num),
          latest_pos_samples(sample_keep_num),
          latest_neg_labels(sample_keep_num),
          latest_neg_samples(sample_keep_num)
    {
        for (int i = 0; i < init_labels.size(); i++) {
            if (init_labels[i] > 0.0) {
                latest_pos_labels.push_back(init_labels[i]);
                latest_pos_samples.push_back(init_samples[i]);
            }
            else {
                latest_neg_labels.push_back(init_labels[i]);
                latest_neg_samples.push_back(init_samples[i]);
            }
        }

        //		assert(!latest_pos_labels.empty() && !latest_neg_labels.empty());

        selectors.reserve(selector_num);
        for (int i = 0; i < selector_num; i++) {
            selectors.push_back(std::make_shared<Selector>(weak_classifier_num));
            for (int j = 0; j < weak_classifier_num; j++) {
                std::shared_ptr<WeakClassifier<Input>> g = generator->generate(init_labels, init_samples);
                selectors[i]->push(g);
            }
        }

        replace_random = std::bind(std::uniform_int_distribution<>(0, replace_cycle - 1), std::ref(mt));
    }


    void update(double label, const Input& x) {
        if (label > 0.0) {
            latest_pos_labels.push_back(label);
            latest_pos_samples.push_back(x);
        }
        else {
            latest_neg_labels.push_back(label);
            latest_neg_samples.push_back(x);
        }

        latest_labels.clear();
        latest_samples.clear();
        latest_labels.reserve(latest_pos_labels.size() + latest_neg_labels.size());
        latest_samples.reserve(latest_pos_samples.size() + latest_neg_samples.size());
        for (int i = 0; i < latest_pos_labels.size(); i++) {
            latest_labels.push_back(latest_pos_labels[i]);
            latest_samples.push_back(latest_pos_samples[i]);
        }
        for (int i = 0; i < latest_neg_labels.size(); i++) {
            latest_labels.push_back(latest_neg_labels[i]);
            latest_samples.push_back(latest_neg_samples[i]);
        }

        double lambda = std::abs(label);
        for (int i = 0; i < selectors.size(); i++) {
            lambda = selectors[i]->update(label * lambda, x);
            if (replace_random() <= 0) {
                std::shared_ptr<WeakClassifier<Input>> weak_classifier = generator->generate(latest_labels, latest_samples);
                selectors[i]->replace(weak_classifier);
            }
            if (lambda < 0.0) {
                std::cout << "break at selector[" << i << "]" << std::endl;
                std::shared_ptr<WeakClassifier<Input>> weak_classifier = generator->generate(latest_labels, latest_samples);
                selectors[i]->replace(weak_classifier);
                break;
            }
        }
    }

    double predictReal(const Input& x) {
        double accum = 0.0;
        for (int i = 0; i < selectors.size(); i++) {
            accum += selectors[i]->predict(x);
        }

        const double upper_bound = 0.5 * std::log((1.0 - 0.05) / 0.05) * selectors.size();
        const double sigmoid_gain = 3.0 / upper_bound;
        double confidence = (1.0 / (1.0 + std::exp(-sigmoid_gain * accum))) * 2.0 - 1.0;

        return confidence;
    }


    int predict(const Input& x) {
        return predictReal(x) > 0 ? +1 : -1;
    }

public:
    std::mt19937 mt;
    std::shared_ptr<Generator> generator;				// weak classifier generator

    std::vector<double> latest_labels;
    std::vector<Input> latest_samples;
    boost::circular_buffer<double> latest_pos_labels;	// labels for generate new weak classifier
    boost::circular_buffer<Input> latest_pos_samples;	// samples for ...
    boost::circular_buffer<double> latest_neg_labels;	// labels for generate new weak classifier
    boost::circular_buffer<Input> latest_neg_samples;	// samples for ...

    std::function<int()> replace_random;
    std::vector<std::shared_ptr<Selector>> selectors;	//
};

}
}
#endif
