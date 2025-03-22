// #include <iostream>
// #include <vector>
// #include <string>
// #include <cmath>
// #include <limits>
// #include <algorithm>
// #include <filesystem>
// #include <opencv2/opencv.hpp>
// #include <opencv2/objdetect.hpp>
// #include <opencv2/imgproc.hpp>

// using namespace std;
// namespace fs = std::filesystem;

// // Class for loading and storing pre-computed features and labels
// class FeatureDatabase {
// private:
//     vector<vector<float>> features;
//     vector<int> labels;

// public:
//     bool load(const string& features_path, const string& labels_path) {
//         cv::FileStorage fs_features(features_path, cv::FileStorage::READ);
//         cv::FileStorage fs_labels(labels_path, cv::FileStorage::READ);
        
//         if (!fs_features.isOpened() || !fs_labels.isOpened()) {
//             cerr << "Error: Could not open feature or label files" << endl;
//             return false;
//         }
        
//         cv::Mat features_mat, labels_mat;
//         fs_features["features"] >> features_mat;
//         fs_labels["labels"] >> labels_mat;
        
//         fs_features.release();
//         fs_labels.release();
        
//         // Convert matrices to vectors
//         features.clear();
//         labels.clear();
        
//         for (int i = 0; i < features_mat.rows; i++) {
//             vector<float> feature_vector;
//             for (int j = 0; j < features_mat.cols; j++) {
//                 feature_vector.push_back(features_mat.at<float>(i, j));
//             }
//             features.push_back(feature_vector);
//             labels.push_back(static_cast<int>(labels_mat.at<float>(i, 0)));
//         }
        
//         cout << "Loaded " << features.size() << " feature vectors and " << labels.size() << " labels" << endl;
//         return true;
//     }
    
//     const vector<vector<float>>& getFeatures() const { return features; }
//     const vector<int>& getLabels() const { return labels; }
// };

// // Feature extraction class
// class FeatureExtractor {
// private:
//     cv::HOGDescriptor hog;
//     cv::Size win_size;
    
// public:
//     FeatureExtractor() : win_size(64, 64) {
//         // Initialize HOG descriptor with the same parameters as in Python
//         cv::Size block_size(16, 16);
//         cv::Size block_stride(8, 8);
//         cv::Size cell_size(8, 8);
//         int nbins = 9;
        
//         hog.winSize = win_size;
//         hog.blockSize = block_size;
//         hog.blockStride = block_stride;
//         hog.cellSize = cell_size;
//         hog.nbins = nbins;
//     }
    
//     vector<float> extractFeatures(const cv::Mat& image) {
//         // Convert to grayscale if needed
//         cv::Mat gray;
//         if (image.channels() == 3) {
//             cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
//         } else {
//             gray = image.clone();
//         }
        
//         // Resize to window size
//         cv::resize(gray, gray, win_size);
        
//         // Compute HOG descriptor
//         vector<float> hog_descriptor;
//         hog.compute(gray, hog_descriptor);
        
//         // Compute Hu Moments
//         cv::Moments moments = cv::moments(gray);
//         double hu_moments[7];
//         cv::HuMoments(moments, hu_moments);
        
//         // Apply -sign(hu) * log10(abs(hu) + 1e-10) as in Python
//         vector<float> hu_features;
//         for (int i = 0; i < 7; i++) {
//             float value = -std::copysign(1.0, hu_moments[i]) * 
//                           std::log10(std::abs(hu_moments[i]) + 1e-10);
//             hu_features.push_back(static_cast<float>(value));
//         }
        
//         // Combine features (HOG + Hu Moments)
//         vector<float> feature_vector = hog_descriptor;
//         feature_vector.insert(feature_vector.end(), hu_features.begin(), hu_features.end());
        
//         return feature_vector;
//     }
    
//     cv::Size getWinSize() const {
//         return win_size;
//     }
// };

// // Screw classifier class
// class ScrewClassifier {
// private:
//     const vector<vector<float>>& features;
//     const vector<int>& labels;
//     float threshold;
    
// public:
//     ScrewClassifier(const vector<vector<float>>& features, const vector<int>& labels, float threshold = 50.0) 
//         : features(features), labels(labels), threshold(threshold) {}
    
//     pair<int, float> classify(const vector<float>& roi_features) {
//         float min_distance = std::numeric_limits<float>::infinity();
//         int best_label = 0;  // Default to non-screw
        
//         for (size_t i = 0; i < features.size(); i++) {
//             // Calculate Euclidean distance
//             float distance = 0.0;
//             for (size_t j = 0; j < min(roi_features.size(), features[i].size()); j++) {
//                 float diff = roi_features[j] - features[i][j];
//                 distance += diff * diff;
//             }
//             distance = sqrt(distance);
            
//             if (distance < min_distance) {
//                 min_distance = distance;
//                 best_label = labels[i];
//             }
//         }
        
//         return make_pair(best_label, min_distance);
//     }
    
//     bool isScrew(const vector<float>& roi_features) {
//         auto [label, distance] = classify(roi_features);
//         return (label == 1 && distance < threshold);
//     }
    
//     void setThreshold(float new_threshold) {
//         threshold = new_threshold;
//     }
// };

// // Screw detector class
// class ScrewDetector {
// private:
//     FeatureExtractor extractor;
//     ScrewClassifier classifier;
    
// public:
//     ScrewDetector(const FeatureDatabase& db, float threshold = 50.0) 
//         : classifier(db.getFeatures(), db.getLabels(), threshold) {}
    
//     void detect(const string& image_path) {
//         // Load image
//         cv::Mat image = cv::imread(image_path);
//         if (image.empty()) {
//             cerr << "[ERROR] Failed to load image: " << image_path << endl;
//             return;
//         }
        
//         // Process image
//         cv::Mat gray;
//         cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
//         // Apply Canny edge detection
//         cv::Mat edges;
//         cv::Canny(gray, edges, 50, 150);
        
//         // Find contours
//         vector<vector<cv::Point>> contours;
//         cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
//         // Process each contour
//         cv::Size min_size = extractor.getWinSize();
        
//         for (const auto& contour : contours) {
//             cv::Rect bbox = cv::boundingRect(contour);
            
//             // Skip ROIs that are too small
//             if (bbox.width < min_size.width || bbox.height < min_size.height) {
//                 continue;
//             }
            
//             // Extract ROI
//             cv::Mat roi = image(bbox);
//             if (roi.empty()) {
//                 continue;
//             }
            
//             // Extract features
//             vector<float> roi_features = extractor.extractFeatures(roi);
            
//             // Classify ROI
//             if (classifier.isScrew(roi_features)) {
//                 // Draw bounding box and label for detected screws
//                 cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 2);
//                 cv::putText(image, "Screw", cv::Point(bbox.x, bbox.y - 10), 
//                             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//             }
//         }
        
//         // Show result
//         cv::namedWindow("Detected Screws", cv::WINDOW_NORMAL);
//         cv::imshow("Detected Screws", image);
//         cv::waitKey(0);
//         cv::destroyAllWindows();
//     }
    
//     // Process a single image without visualization (for batch processing)
//     cv::Mat detectWithoutDisplay(const cv::Mat& image) {
//         cv::Mat result = image.clone();
        
//         // Process image
//         cv::Mat gray;
//         cv::cvtColor(result, gray, cv::COLOR_BGR2GRAY);
        
//         // Apply Canny edge detection
//         cv::Mat edges;
//         cv::Canny(gray, edges, 50, 150);
        
//         // Find contours
//         vector<vector<cv::Point>> contours;
//         cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
//         // Process each contour
//         cv::Size min_size = extractor.getWinSize();
        
//         for (const auto& contour : contours) {
//             cv::Rect bbox = cv::boundingRect(contour);
            
//             // Skip ROIs that are too small
//             if (bbox.width < min_size.width || bbox.height < min_size.height) {
//                 continue;
//             }
            
//             // Extract ROI
//             cv::Mat roi = result(bbox);
//             if (roi.empty()) {
//                 continue;
//             }
            
//             // Extract features
//             vector<float> roi_features = extractor.extractFeatures(roi);
            
//             // Classify ROI
//             if (classifier.isScrew(roi_features)) {
//                 // Draw bounding box and label for detected screws
//                 cv::rectangle(result, bbox, cv::Scalar(0, 255, 0), 2);
//                 cv::putText(result, "Screw", cv::Point(bbox.x, bbox.y - 10), 
//                             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//             }
//         }
        
//         return result;
//     }
    
//     // Set classification threshold
//     void setThreshold(float threshold) {
//         classifier.setThreshold(threshold);
//     }
// };

// // Save features and labels to files
// void saveFeaturesToFile(const vector<vector<float>>& features, const vector<int>& labels,
//                         const string& features_path, const string& labels_path) {
//     // Prepare features matrix
//     int num_samples = features.size();
//     int num_features = features[0].size();
//     cv::Mat features_mat(num_samples, num_features, CV_32F);
    
//     for (int i = 0; i < num_samples; i++) {
//         for (int j = 0; j < num_features; j++) {
//             features_mat.at<float>(i, j) = features[i][j];
//         }
//     }
    
//     // Prepare labels matrix
//     cv::Mat labels_mat(num_samples, 1, CV_32F);
//     for (int i = 0; i < num_samples; i++) {
//         labels_mat.at<float>(i, 0) = static_cast<float>(labels[i]);
//     }
    
//     // Save to files
//     cv::FileStorage fs_features(features_path, cv::FileStorage::WRITE);
//     fs_features << "features" << features_mat;
//     fs_features.release();
    
//     cv::FileStorage fs_labels(labels_path, cv::FileStorage::WRITE);
//     fs_labels << "labels" << labels_mat;
//     fs_labels.release();
    
//     cout << "Saved " << num_samples << " feature vectors and labels to files" << endl;
// }

// int main(int argc, char** argv) {
//     // Check if image path is provided
//     if (argc < 2) {
//         cout << "Usage: " << argv[0] << " <image_path> [threshold]" << endl;
//         return -1;
//     }
    
//     string image_path = argv[1];
//     float threshold = 50.0;  // Default threshold
    
//     if (argc >= 3) {
//         threshold = stof(argv[2]);
//     }
    
//     // Load pre-computed features and labels
//     FeatureDatabase db;
//     if (!db.load("models/features.xml", "models/labels.xml")) {
//         cerr << "Failed to load feature database. Make sure the files exist." << endl;
        
//         // If this is the first run, we might need to convert the numpy arrays to OpenCV format
//         cout << "Do you want to convert numpy arrays to OpenCV format? (y/n): ";
//         char response;
//         cin >> response;
        
//         if (response == 'y' || response == 'Y') {
//             cout << "Please implement the numpy to OpenCV conversion or run the provided Python script for conversion." << endl;
//             // This would require a Python bridge or separate utility
//         }
        
//         return -1;
//     }
    
//     // Create detector
//     ScrewDetector detector(db, threshold);
    
//     // Detect screws
//     detector.detect(image_path);
    
//     return 0;
// }







#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <svm.h>
#include "feature_extraction.h" // Include the feature extraction header

using namespace std;

// SVM wrapper class (same as in train.cpp)
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
        // Implementation remains the same as in train.cpp
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

// Function to display the detection result
void display_result(const cv::Mat& img, bool is_screw) {
    cv::Mat display_img = img.clone();
    string label = is_screw ? "SCREW" : "NOT SCREW";
    cv::Scalar color = is_screw ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    
    // Add label to the image
    cv::putText(display_img, label, cv::Point(20, 40), 
                cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    
    // Draw border
    cv::rectangle(display_img, cv::Point(0, 0), 
                 cv::Point(display_img.cols-1, display_img.rows-1), color, 3);
    
    // Display the image
    cv::namedWindow("Screw Detection", cv::WINDOW_NORMAL);
    cv::imshow("Screw Detection", display_img);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    // Check if image path is provided
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }
    
    string image_path = argv[1];
    
    // Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        cerr << "Error: Could not load image at " << image_path << endl;
        return -1;
    }
    
    cout << "Loaded image: " << image_path << endl;
    
    // Extract features
    cout << "Extracting features..." << endl;
    vector<float> features = extract_features(img);
    vector<vector<float>> X = {features};
    
    // Load scaler
    cout << "Loading scaler..." << endl;
    StandardScaler scaler;
    scaler.load("scaler.dat");
    
    // Standardize features
    X = scaler.transform(X);
    
    // Load SVM model
    cout << "Loading SVM model..." << endl;
    SVC svm;
    svm.load("model.svmmodel");
    
    // Predict
    cout << "Making prediction..." << endl;
    vector<int> predictions = svm.predict(X);
    
    // Result
    bool is_screw = (predictions[0] == 1);
    if (is_screw) {
        cout << "Result: SCREW DETECTED" << endl;
    } else {
        cout << "Result: NOT A SCREW" << endl;
    }
    
    // Display result
    display_result(img, is_screw);
    
    return 0;
}