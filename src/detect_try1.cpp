#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>  // Include OpenCV's ML module
#include "feature_extraction.h"

using namespace std;

// Function to merge overlapping bounding boxes
vector<cv::Rect> merge_boxes(vector<cv::Rect> boxes) {
    vector<cv::Rect> merged;
    
    while (!boxes.empty()) {
        cv::Rect current = boxes[0];
        boxes.erase(boxes.begin());
        
        int i = 0;
        while (i < boxes.size()) {
            cv::Rect& box = boxes[i];
            if ((current & box).area() > 0) {  // If boxes intersect
                current = current | box;  // Merge boxes (union)
                boxes.erase(boxes.begin() + i);
            } else {
                i++;
            }
        }
        merged.push_back(current);
    }
    
    return merged;
}

// Function to remove nested bounding boxes
vector<cv::Rect> remove_nested_boxes(const vector<cv::Rect>& boxes) {
    vector<cv::Rect> filtered;
    
    for (const auto& box : boxes) {
        bool inside = false;
        for (const auto& other : boxes) {
            if (box != other &&
                box.x > other.x && box.y > other.y &&
                box.x + box.width < other.x + other.width &&
                box.y + box.height < other.y + other.height) {
                inside = true;  // Box is completely inside another
                break;
            }
        }
        if (!inside) {
            filtered.push_back(box);
        }
    }
    
    return filtered;
}

int main() {
    // Load trained model and scaler
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("model.svmmodel");
    StandardScaler scaler;
    
    try {
        scaler.load("scaler.dat");
        if (svm.empty()) {
            cerr << "Error: Failed to load SVM model." << endl;
            return 1;
        }
    } catch (const std::exception& e) {
        cerr << "Error loading model or scaler: " << e.what() << endl;
        return 1;
    }
    
    // Load the test image
    string image_path = "../dataset/test/005.png";  // Change this to your test image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        cerr << "Error: Unable to load image. Check the path." << endl;
        return 1;
    }
    
    // Convert image to grayscale
    cv::Mat gray, blurred, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);  // Reduce noise
    
    // Edge Detection
    cv::Canny(blurred, edges, 50, 150);
    
    // Find Contours
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(edges.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Store bounding boxes
    vector<cv::Rect> bounding_boxes;
    
    // Process each detected object
    cout << "Processing " << contours.size() << " contours..." << endl;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        
        // Skip very small contours
        if (rect.width < 10 || rect.height < 10) continue;
        
        // Make sure the rect is within image boundaries
        rect = rect & cv::Rect(0, 0, image.cols, image.rows);
        if (rect.width <= 0 || rect.height <= 0) continue;
        
        // Extract ROI
        cv::Mat roi = image(rect);
        
        // Extract features
        vector<float> test_features = extract_features(roi);
        
        // Transform features
        vector<vector<float>> test_features_vector = {test_features};
        vector<vector<float>> transformed_features = scaler.transform(test_features_vector);
        
        // Convert to OpenCV Mat for prediction
        cv::Mat features_mat(1, transformed_features[0].size(), CV_32F);
        for (size_t i = 0; i < transformed_features[0].size(); i++) {
            features_mat.at<float>(0, i) = transformed_features[0][i];
        }
        
        // Predict
        float prediction = svm->predict(features_mat);
        
        // If classified as a screw, store bounding box
        if (prediction == 1.0) {
            bounding_boxes.push_back(rect);
            cout << "Screw detected at (" << rect.x << ", " << rect.y << ")" << endl;
        }
    }
    
    // Merge overlapping boxes
    vector<cv::Rect> merged_boxes = merge_boxes(bounding_boxes);
    
    // Remove nested boxes
    vector<cv::Rect> final_boxes = remove_nested_boxes(merged_boxes);
    
    cout << "Found " << final_boxes.size() << " screws after merging and filtering." << endl;
    
    // Draw bounding boxes
    for (const auto& rect : final_boxes) {
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
        cv::putText(image, "Screw", cv::Point(rect.x, rect.y - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    
    // Resize image if needed (max 1280x720)
    cv::Mat image_resized;
    int h = image.rows, w = image.cols;
    if (w > 1280 || h > 720) {
        double scale = min(1280.0 / w, 720.0 / h);
        int new_w = static_cast<int>(w * scale);
        int new_h = static_cast<int>(h * scale);
        cv::resize(image, image_resized, cv::Size(new_w, new_h));
    } else {
        image_resized = image.clone();
    }
    
    // Show image in a window
    cv::imshow("Detected Screws", image_resized);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}