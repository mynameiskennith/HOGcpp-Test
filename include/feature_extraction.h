#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>  // Required for file operations
#include <numeric>

// StandardScaler class for feature normalization
class StandardScaler {
private:
    std::vector<float> mean_;
    std::vector<float> scale_;
    bool fitted_ = false;

public:
    StandardScaler() = default;

    void fit(const std::vector<std::vector<float>>& X) {
        if (X.empty() || X[0].empty()) {
            throw std::runtime_error("Cannot fit empty dataset");
        }

        size_t n_features = X[0].size();
        size_t n_samples = X.size();

        // Initialize mean and scale vectors
        mean_.resize(n_features, 0.0f);
        scale_.resize(n_features, 0.0f);

        // Calculate mean
        for (const auto& sample : X) {
            if (sample.size() != n_features) {
                throw std::runtime_error("Inconsistent number of features");
            }
            
            for (size_t j = 0; j < n_features; ++j) {
                mean_[j] += sample[j];
            }
        }

        for (size_t j = 0; j < n_features; ++j) {
            mean_[j] /= n_samples;
        }

        // Calculate standard deviation
        for (const auto& sample : X) {
            for (size_t j = 0; j < n_features; ++j) {
                float diff = sample[j] - mean_[j];
                scale_[j] += diff * diff;
            }
        }

        for (size_t j = 0; j < n_features; ++j) {
            scale_[j] = std::sqrt(scale_[j] / n_samples);
            // Avoid division by zero
            if (scale_[j] < 1e-10) {
                scale_[j] = 1.0f;
            }
        }

        fitted_ = true;
    }

    std::vector<std::vector<float>> transform(const std::vector<std::vector<float>>& X) const {
        if (!fitted_) {
            throw std::runtime_error("StandardScaler is not fitted yet");
        }

        if (X.empty()) {
            return std::vector<std::vector<float>>();
        }

        size_t n_features = X[0].size();
        size_t n_samples = X.size();

        if (mean_.size() != n_features) {
            throw std::runtime_error("Feature dimension mismatch");
        }

        std::vector<std::vector<float>> X_transformed(n_samples, std::vector<float>(n_features));

        for (size_t i = 0; i < n_samples; ++i) {
            if (X[i].size() != n_features) {
                throw std::runtime_error("Inconsistent number of features");
            }
            
            for (size_t j = 0; j < n_features; ++j) {
                X_transformed[i][j] = (X[i][j] - mean_[j]) / scale_[j];
            }
        }

        return X_transformed;
    }

    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }

        size_t n_features = mean_.size();
        file.write(reinterpret_cast<const char*>(&n_features), sizeof(n_features));
        file.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
        
        file.write(reinterpret_cast<const char*>(mean_.data()), n_features * sizeof(float));
        file.write(reinterpret_cast<const char*>(scale_.data()), n_features * sizeof(float));
        
        file.close();
    }

    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }

        size_t n_features;
        file.read(reinterpret_cast<char*>(&n_features), sizeof(n_features));
        file.read(reinterpret_cast<char*>(&fitted_), sizeof(fitted_));
        
        mean_.resize(n_features);
        scale_.resize(n_features);
        
        file.read(reinterpret_cast<char*>(mean_.data()), n_features * sizeof(float));
        file.read(reinterpret_cast<char*>(scale_.data()), n_features * sizeof(float));
        
        file.close();
    }
};

// Extract HOG features from an image
std::vector<float> extract_features(const cv::Mat& img) {
    // Resize image to consistent dimensions
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(64, 64));
    
    // Convert to grayscale if it's a color image
    cv::Mat gray;
    if (resized.channels() == 3) {
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = resized.clone();
    }
    
    // Initialize HOG descriptor
    cv::HOGDescriptor hog(
        cv::Size(64, 64),    // Window size
        cv::Size(16, 16),    // Block size
        cv::Size(8, 8),      // Block stride
        cv::Size(8, 8),      // Cell size
        9                    // Number of bins
    );
    
    // Compute HOG features
    std::vector<float> descriptors;
    std::vector<cv::Point> locations;
    hog.compute(gray, descriptors, cv::Size(0, 0), cv::Size(0, 0), locations);
    
    return descriptors;
}

#endif // FEATURE_EXTRACTION_H