#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <svm.h>
#include "feature_extraction.h" // Include the feature extraction header

using namespace std;
namespace fs = std::filesystem;

// Load dataset
vector<cv::Mat> load_images_from_folder(const string& folder) {
    vector<cv::Mat> images;
    for (const auto& entry : fs::directory_iterator(folder)) {
        string img_path = entry.path().string();
        cv::Mat img = cv::imread(img_path);
        if (!img.empty()) {
            images.push_back(img);
        }
    }
    return images;
}

// SVM wrapper class
class SVC {
private:
    svm_parameter param;
    svm_problem prob;
    svm_model* model = nullptr;
    bool probability;

public:
    SVC(const string& kernel_type = "rbf", bool probability_estimates = false) {
        // Set default parameters
        param.svm_type = C_SVC;
        param.kernel_type = RBF;
        if (kernel_type == "linear") param.kernel_type = LINEAR;
        else if (kernel_type == "poly") param.kernel_type = POLY;
        else if (kernel_type == "sigmoid") param.kernel_type = SIGMOID;
        
        param.degree = 3;
        param.gamma = 0; // 1/num_features
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 1;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = probability_estimates ? 1 : 0;
        param.nr_weight = 0;
        param.weight_label = nullptr;
        param.weight = nullptr;
        
        this->probability = probability_estimates;
    }
    
    ~SVC() {
        if (model) svm_free_and_destroy_model(&model);
        svm_destroy_param(&param);
    }
    
    void fit(const vector<vector<float>>& X, const vector<int>& y) {
        if (X.empty() || y.empty() || X.size() != y.size()) {
            throw runtime_error("Invalid input data");
        }
        
        // Set gamma if not specified
        if (param.gamma == 0) param.gamma = 1.0 / X[0].size();
        
        // Prepare problem
        prob.l = X.size();
        prob.y = new double[prob.l];
        prob.x = new svm_node*[prob.l];
        
        // Fill in the problem
        for (int i = 0; i < prob.l; i++) {
            prob.y[i] = y[i];
            
            // +1 for the -1 index terminator
            prob.x[i] = new svm_node[X[i].size() + 1];
            
            for (size_t j = 0; j < X[i].size(); j++) {
                prob.x[i][j].index = j + 1;  // LIBSVM uses 1-based indexing
                prob.x[i][j].value = X[i][j];
            }
            
            // Terminate with -1 index
            prob.x[i][X[i].size()].index = -1;
        }
        
        // Train the model
        model = svm_train(&prob, &param);
    }
    
    vector<int> predict(const vector<vector<float>>& X) {
        if (!model) throw runtime_error("Model not trained yet");
        
        vector<int> predictions;
        predictions.reserve(X.size());
        
        for (const auto& sample : X) {
            // Convert sample to svm_node format
            svm_node* x = new svm_node[sample.size() + 1];
            for (size_t j = 0; j < sample.size(); j++) {
                x[j].index = j + 1;  // LIBSVM uses 1-based indexing
                x[j].value = sample[j];
            }
            x[sample.size()].index = -1;
            
            // Predict
            double prediction = svm_predict(model, x);
            predictions.push_back(static_cast<int>(prediction));
            
            delete[] x;
        }
        
        return predictions;
    }
    
    void save(const string& filename) {
        if (!model) throw runtime_error("Model not trained yet");
        svm_save_model(filename.c_str(), model);
    }
    
    void load(const string& filename) {
        model = svm_load_model(filename.c_str());
        if (!model) throw runtime_error("Failed to load model from " + filename);
    }
};

int main() {
    // Load screw and non-screw images
    vector<cv::Mat> screw_images = load_images_from_folder("../dataset/screws/");
    vector<cv::Mat> non_screw_images = load_images_from_folder("../dataset/non_screws/");
    
    cout << "Loaded " << screw_images.size() << " screw images and " 
         << non_screw_images.size() << " non-screw images." << endl;

    // Assign labels (1 for screws, 0 for non-screws)
    vector<int> screw_labels(screw_images.size(), 1);
    vector<int> non_screw_labels(non_screw_images.size(), 0);

    cout << "Extracting features..." << endl;
    // Extract features
    vector<vector<float>> screw_features, non_screw_features;
    for (const auto& img : screw_images) {
        screw_features.push_back(extract_features(img));
    }
    for (const auto& img : non_screw_images) {
        non_screw_features.push_back(extract_features(img));
    }

    // Create dataset
    vector<vector<float>> X(screw_features);
    X.insert(X.end(), non_screw_features.begin(), non_screw_features.end());
    vector<int> y;
    y.insert(y.end(), screw_labels.begin(), screw_labels.end());
    y.insert(y.end(), non_screw_labels.begin(), non_screw_labels.end());

    cout << "Standardizing features..." << endl;
    // Standardize features
    StandardScaler scaler;
    scaler.fit(X);
    X = scaler.transform(X);

    cout << "Training SVM classifier..." << endl;
    // Train an SVM classifier
    SVC svm("linear", true);
    svm.fit(X, y);

    cout << "Saving model and scaler..." << endl;
    // Save model and scaler
    svm.save("model.svmmodel");
    scaler.save("scaler.dat");

    cout << "Training complete. Model saved as 'model.svmmodel'." << endl;

    return 0;
}