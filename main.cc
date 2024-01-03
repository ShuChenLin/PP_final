#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>

using namespace std;

class SoftMarginSVM {
public:
    SoftMarginSVM(double C, double tolerance, int maxIterations)
        : C(C), tolerance(tolerance), maxIterations(maxIterations) {}

    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
        int numSamples = data.size();
        int numFeatures = data[0].size();

        // Initialize alpha and bias
        std::vector<double> alpha(numSamples, 0.0);
        double bias = 0.0;

        // Training loop
        int iterations = 0;
        while (iterations < maxIterations) {
            int numChangedAlphas = 0;

            for (int i = 0; i < numSamples; ++i) {
                double error_i = calculateError(i, data, labels, alpha, bias);

                if ((labels[i] * error_i < -tolerance && alpha[i] < C) ||
                    (labels[i] * error_i > tolerance && alpha[i] > 0)) {
                    int j = selectSecondAlpha(i, numSamples);

                    double error_j = calculateError(j, data, labels, alpha, bias);
                    double old_alpha_i = alpha[i];
                    double old_alpha_j = alpha[j];

                    double L, H;
                    if (labels[i] != labels[j]) {
                        L = std::max(0.0, alpha[j] - alpha[i]);
                        H = std::min(C, C + alpha[j] - alpha[i]);
                    } else {
                        L = std::max(0.0, alpha[i] + alpha[j] - C);
                        H = std::min(C, alpha[i] + alpha[j]);
                    }

                    if (L == H) continue;

                    double eta = 2.0 * kernel(data[i], data[j]) - kernel(data[i], data[i]) -
                                 kernel(data[j], data[j]);
                    if (eta >= 0) continue;

                    alpha[j] = alpha[j] - (labels[j] * (error_i - error_j)) / eta;
                    alpha[j] = std::min(std::max(alpha[j], L), H);

                    if (std::abs(alpha[j] - old_alpha_j) < 1e-5) continue;

                    alpha[i] = alpha[i] + labels[i] * labels[j] * (old_alpha_j - alpha[j]);

                    double b1 = bias - error_i - labels[i] * (alpha[i] - old_alpha_i) * kernel(data[i], data[i]) -
                                labels[j] * (alpha[j] - old_alpha_j) * kernel(data[i], data[j]);
                    double b2 = bias - error_j - labels[i] * (alpha[i] - old_alpha_i) * kernel(data[i], data[j]) -
                                labels[j] * (alpha[j] - old_alpha_j) * kernel(data[j], data[j]);

                    if (alpha[i] > 0 && alpha[i] < C) {
                        bias = b1;
                    } else if (alpha[j] > 0 && alpha[j] < C) {
                        bias = b2;
                    } else {
                        bias = (b1 + b2) / 2.0;
                    }

                    numChangedAlphas++;
                }
            }

            if (numChangedAlphas == 0) {
                iterations++;
            } else {
                iterations = 0;
            }
        }

        // Store the model parameters
        this->alpha = alpha;
        this->bias = bias;

        // Find support vectors
        for (int i = 0; i < numSamples; ++i) {
            if (alpha[i] > 0) {
                supportVectors.push_back(data[i]);
                supportVectorLabels.push_back(labels[i]);
            }
        }
    }

    int predict(const std::vector<double>& dataPoint) {
        double result = bias;
        for (size_t i = 0; i < supportVectors.size(); ++i) {
            result += alpha[i] * supportVectorLabels[i] * kernel(supportVectors[i], dataPoint);
        }
        return (result >= 0) ? 1 : 2;
    }

private:
    double C;
    double tolerance;
    int maxIterations;
    std::vector<double> alpha;
    double bias;
    std::vector<std::vector<double>> supportVectors;
    std::vector<int> supportVectorLabels;

    double calculateError(int index, const std::vector<std::vector<double>>& data, 
                          const std::vector<int>& labels, const std::vector<double>& alpha, double bias) {
        double result = bias;
        for (size_t i = 0; i < data.size(); ++i) {
            result += alpha[i] * labels[i] * kernel(data[index], data[i]);
        }
        return result - labels[index];
    }

    int selectSecondAlpha(int firstAlphaIndex, int numSamples) {
        int secondAlphaIndex = firstAlphaIndex;
        while (secondAlphaIndex == firstAlphaIndex) {
            secondAlphaIndex = rand() % numSamples;
        }
        return secondAlphaIndex;
    }

    double kernel(const std::vector<double>& x1, const std::vector<double>& x2) {
        // This example uses a simple linear kernel
        double result = 0.0;
        for (size_t i = 0; i < x1.size(); ++i) {
            result += x1[i] * x2[i];
        }
        return result;
    }
};

int main() {
    // std::vector<std::vector<double>> data = {{2.0, 1.0}, {3.0, 4.0}, {1.0, 3.0}, {3.5, 2.0}};
    // std::vector<int> labels = {1, 1, -1, -1};

    // read data from file

    std::vector<std::vector<double>> data;
    std::vector<int> labels;

    std::ifstream file("data.in");
    std::string line;

    // first 16 element in a row are features, last one is label
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double value, tmp;
        while (iss >> value) {
            row.push_back(value);
            tmp = value;
        }
        labels.push_back(int(value));
        row.pop_back();
        data.push_back(row);
    }

    // shuffle data into training and testing set
    std::vector<std::vector<double>> training_data;
    std::vector<int> training_labels;

    std::vector<std::vector<double>> testing_data;
    std::vector<int> testing_labels;

    // 10% of data is used for testing
    int testing_size = data.size() * 20 / 100;

    for (int i = 0; i < data.size(); ++i) {
        if (i < testing_size) {
            testing_data.push_back(data[i]);
            testing_labels.push_back(labels[i]);
        } else {
            training_data.push_back(data[i]);
            training_labels.push_back(labels[i]);
        }
    }

    // Create a SoftMarginSVM model and train it
    SoftMarginSVM svm(5, 0.0001, 10000);
    svm.train(training_data, training_labels);

    // Make predictions
    vector<int> predictions;
    for (int i = 0; i < testing_data.size(); ++i) {
        predictions.push_back(svm.predict(testing_data[i]));
    }

    // Make predictions with training data

    vector<int> predictions_training;
    for (int i = 0; i < training_data.size(); ++i) {
        predictions_training.push_back(svm.predict(training_data[i]));
    }

    // cal accuracy
    int correct = 0;
    for (int i = 0; i < testing_data.size(); ++i) {
        //cout << "prediction: " << predictions[i] << " label: " << testing_labels[i] << endl;
        if (predictions[i] == testing_labels[i]) {
            correct++;
        }
    }

    int correct_training = 0;
    for (int i = 0; i < training_data.size(); ++i) {
        cout << "prediction: " << predictions_training[i] << " label: " << training_labels[i] << endl;
        if (predictions_training[i] == training_labels[i]) {
            correct_training++;
        }
    }

    std::cout << "Accuracy: " << (double)correct / testing_data.size() << std::endl;

    std::cout << "Accuracy training: " << (double)correct_training / training_data.size() << std::endl;

    return 0;
}

