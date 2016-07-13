#include "kNN.h"

typedef std::vector<std::vector<double>> Matrix;

int main(int argc, char const *argv[])
{
    const std::vector<double> firstCoordinate = {5, 5};
    const std::vector<double> secondCoordinate = {10, 10};
    const std::vector<double> thirdCoordinate = {15, 15};
    const std::vector<double> forthCoordinate = {20, 20};
    const Matrix trainingMatrix = {firstCoordinate, secondCoordinate, thirdCoordinate, forthCoordinate};
    const std::vector<int> labels = {1, 1, 3, 3};
    KNeighborsClassifier testKnn(3);
    testKnn.Fit(trainingMatrix, labels);
    Matrix predictMatrix = {std::vector<double> {7, 7}, std::vector<double> {17, 17}};
    std::vector<int> output = testKnn.Predict(predictMatrix);
    for (int i = 0; i < output.size(); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << "\n";
    return 0;
}