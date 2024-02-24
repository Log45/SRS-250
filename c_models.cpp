#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

using namespace std;

double sigmoid(double x){
    return (1.0f / (1.0f + std::exp(-x)));
}

double linear(double x, double weight, double bias){
    return ((weight * x) + bias);
}

class LinearRegressionModel{
    public:
        double weight;
        double bias;
        LinearRegressionModel(){
            weight = 2 * rand() / RAND_MAX -1;
            bias = 2 * rand() / RAND_MAX -1;
        }

        double forward(double x){
            return linear(x, weight, bias);
        }
};

class ClassificationModel{
    public:
        int N; // number of neurons in input layer
        int H; // number of neurons in hidden layer
        int O; // number of output features
        double ** weights;
        double ** bias;
        ClassificationModel(int in_features, int out_features, int hidden_units=8){
            N = in_features; // columns
            H = hidden_units; // hidden rows, output columns
            O = out_features; // output rows
            // TODO: Instanciate random weights and biases based on the input and output features
        }

        double forward(double ** x){

        }
};