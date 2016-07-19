#include "kNN.h"

static const int thisIsZero = 0;

KNeighborsClassifier::KNeighborsClassifier(size_t kneighbors): KNeighbors(kneighbors) {}

void KNeighborsClassifier::CheckInputFit(const Matrix& neighborCoordinates, const std::vector<int>& neighborLabels) const {
    if (neighborCoordinates.size() != neighborLabels.size()) {
        std::cerr << "bad input\n" << "neighborCoordinates.size() == " << neighborCoordinates.size() 
                    << " neighborLabels.size() == " << neighborLabels.size() << "\n";
        exit(1);
    }
    if (KNeighbors > neighborLabels.size()) {
        std::cerr << "bad input\n" << "KNeighbors == " << KNeighbors
                    << " coordinateMatrix.size() == " << neighborLabels.size() << "\n";
        exit(1);
    }
    CheckMatrixRectangular(neighborCoordinates);
}

void CheckMatrixRectangular(const Matrix& matrixChecked) {
    for (size_t i = 0; i < matrixChecked.size() - 1; ++i) {
        if (matrixChecked[i].size() != matrixChecked[i + 1].size()) {
            std::cerr << "Matrix is not rectangular\n" << "matrixChecked[i].size() == " << matrixChecked[i].size()
                        << " matrixChecked[i].size() == " << matrixChecked[i + 1].size() << "\n";
            exit(1);
        }
    }
}

void KNeighborsClassifier::Fit(const Matrix& neighborCoordinates, const std::vector<int>& neighborLabels) {
    CheckInputFit(neighborCoordinates, neighborLabels);
    coordinateMatrix = neighborCoordinates;
    labels = neighborLabels;
}

double MeasureDistanceCoordinates(const std::vector<double>& firstCoordinate, const std::vector<double>& secondCoordinate) {
    double distanceSquared = 0;
        for (int j = 0; j < firstCoordinate.size(); ++j) {
            distanceSquared += pow(firstCoordinate[j] - secondCoordinate[j], 2);
        }
    return sqrt(distanceSquared);
}

void KNeighborsClassifier::CheckInputPredict(const Matrix& matrixToPredict) const {
    CheckMatrixRectangular(matrixToPredict);
    if (coordinateMatrix[thisIsZero].size() != matrixToPredict[thisIsZero].size()) {
        std::cerr << "different dimensions\n" << "coordinateMatrix[thisIsZero].size() == " << coordinateMatrix[thisIsZero].size()
                    << " matrixToPredict[thisIsZero].size() == " << matrixToPredict[thisIsZero].size();
        exit(1);
    }
}

int KNeighborsClassifier::PredictSingleCoordinate(const std::vector<double>& vectorToPredict) const {
    std::priority_queue<std::pair<double, size_t>> queueDistanceLabelPair;
    int i = 0;
    for ( ; i < KNeighbors; ++i) {
        queueDistanceLabelPair.push(std::make_pair(MeasureDistanceCoordinates(vectorToPredict, coordinateMatrix[i]), labels[i]));
    }

    for ( ; i < coordinateMatrix.size(); ++i) {
        double distanceOfIthString = MeasureDistanceCoordinates(vectorToPredict, coordinateMatrix[i]);
        if (distanceOfIthString < queueDistanceLabelPair.top().first) {
            queueDistanceLabelPair.pop();
            queueDistanceLabelPair.push(std::make_pair(distanceOfIthString, labels[i]));
        }
    }

    std::unordered_map<size_t, size_t> labelsCount;
    while (!queueDistanceLabelPair.empty()) {
        ++labelsCount[queueDistanceLabelPair.top().second];
        queueDistanceLabelPair.pop();
    }

    size_t PredictedLabel = labelsCount.begin()->first;
    size_t labelMaxFrequency = labelsCount.begin()->second;
    for (const auto& x : labelsCount) {
        if (x.second > labelMaxFrequency) {
            PredictedLabel = x.first;
            labelMaxFrequency = x.second;
        }
    }

    return PredictedLabel;
}

std::vector<int> KNeighborsClassifier::Predict(const Matrix& matrixToPredict) const {
    CheckInputPredict(matrixToPredict);
    std::vector<int> output;
    output.reserve(matrixToPredict.size());
    for (const auto& x : matrixToPredict) {
        output.push_back(PredictSingleCoordinate(x));
    }
    return output;
}