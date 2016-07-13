#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <algorithm>

typedef std::vector<std::vector<double>> Matrix;

class KNeighborsClassifier {
private:
    size_t KNeighbors;
    Matrix coordinateMatrix;
    std::vector<int> labels;
    void CheckInputFit(const Matrix& neighborCoordinates, const std::vector<int>& neighborLabels) const;
    void CheckInputPredict(const Matrix& matrixToPredict) const;
    int PredictSingleCoordinate(const std::vector<double>& vectorToPredict) const;
public:
    KNeighborsClassifier(size_t);
    void Fit(const Matrix&, const std::vector<int>&);
    std::vector<int> Predict(const Matrix& matrixToPredict) const;
};

void CheckMatrixRectangular(const Matrix& matrixChecked);
double MeasureDistanceCoordinates(const std::vector<double>& firstCoordinate, const std::vector<double>& secondCoordinate);