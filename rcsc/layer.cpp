#include <iostream>
#include "layer.h"
#include "utils/utils.h"
using namespace utils;
namespace rcsc {

std::vector<Matrix&> Layer::get_params()
{
    std::vector<Matrix&> layer_param;
    layer_param.push_back(*weight);
    layer_param.push_back(*bias);
    return layer_param;
}

void Layer::clear_grad()
{
    if (weight_grad != nullptr)
        weight_grad->clear();
    if (bias_grad != nullptr)
        bias_grad->clear();
}

Linear::Linear(int input_size, int output_size){
    weight = new Matrix(output_size,input_size);
    bias = new Matrix(output_size);
}

Matrix Linear::forward(Matrix & input){
    *this->input = input;
    return Matrix::times(*weight,*this->input) + *bias;
}

Matrix Linear::backward(Matrix &gradient){
    *weight_grad += Matrix::times(*input,gradient,true);
    *bias_grad += gradient;
    return Matrix::times(gradient,*weight,true);
}

Matrix Sigmod::forward(Matrix &input) {
    return input.activation(utils::sigmod);
}

Matrix Sigmod::backward(Matrix &gardient){
    return gardient.derivation(utils::deltaSigmod);
}

Matrix ReLU::forward(Matrix &input) {
    *activated = input.activation(utils::relu) > 0;
    return *activated;
}

Matrix ReLU::backward(Matrix &gradient) {
    return Matrix::times(gradient,*activated) > 0;
}

}
