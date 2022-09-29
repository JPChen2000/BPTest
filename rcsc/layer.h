#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include"utils/matrix.h"

using namespace utils;
namespace rcsc {

class Layer{
public:
    Layer() { };
    Matrix forward(Matrix &input);
    Matrix backward(Matrix &gradient);
    std::vector<Matrix &> get_params();
    void clear_grad();
protected:
    Matrix *weight;
    Matrix *weight_grad;
    Matrix *bias;
    Matrix *bias_grad;
    Matrix *input;
};

class Linear : public Layer{
public:
    Linear() = delete;
    Linear(int input_size,int output_size);

    Matrix forward(Matrix &input);
    Matrix backward(Matrix &gradient);

};

class ReLU : public Layer{
public:
    ReLU();
    Matrix forward(Matrix &input);
    Matrix backward(Matrix &gradient);
protected:
    Matrix *activated;
};

class Sigmod : public Layer {
public:
    Sigmod();
    Matrix forward(Matrix &input);
    Matrix backward(Matrix &gardient);
};

}


#endif // LAYER_H
