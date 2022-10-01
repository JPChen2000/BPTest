#include <iostream>
#include "rcsc/layer.h"
#include "utils/utils.h"
using namespace utils;
namespace rcsc {

std::vector<Matrix *> Layer::get_params()
{
    std::vector<Matrix *> layer_param;
    layer_param.push_back(weight);
    layer_param.push_back(bias);
    return layer_param;
}

std::vector<Matrix *> Layer::get_grads()
{
    std::vector<Matrix *> layer_grads;
    layer_grads.push_back(weight_grad);
    layer_grads.push_back(bias_grad);
    return layer_grads;
}

std::vector<Matrix *> Layer::get_veloc()
{
    std::vector<Matrix *> layer_veloc;
    layer_veloc.push_back(veloc_weight_grad);
    layer_veloc.push_back(veloc_bias_grad);
    return layer_veloc;
}

void Layer::clear_grad()
{
    if (weight_grad != nullptr)
        weight_grad->clear();
    if (bias_grad != nullptr)
        bias_grad->clear();
}

bool Layer::is_activate_layer()
{
    if (weight == nullptr)
    {
        return true;
    }
    return false;
}

void Layer::print()
{
    std::cout << "layer:" << m_name << std::endl;
}

Linear::Linear(std::string name, int input_size, int output_size){
    m_name = name;
    weight = new Matrix(input_size, output_size);
    weight_grad = new Matrix(input_size, output_size);
    bias = new Matrix(1,output_size);
    bias_grad = new Matrix(1,output_size);
    veloc_weight_grad = new Matrix(input_size, output_size, true);
    veloc_bias_grad = new Matrix(1, output_size, true);
}

Matrix Linear::forward(Matrix & input){
    this->input = input;
    return Matrix::times(input, *weight) + *bias;
}

Matrix Linear::backward(Matrix &gradient){
    Matrix input_T = input.rotate();
    Matrix weight_grad_delta = Matrix::times(input_T ,gradient);
    *weight_grad =  *weight_grad + weight_grad_delta;
    *bias_grad = gradient.sum_axis_0() + *bias_grad;
    Matrix weight_T = weight->rotate();
    return Matrix::times(gradient, weight_T);
}



void Linear::print()
{
    std::cout << "layer:" << m_name << " ";
    weight->printSize();
    std::cout << std::endl;
    bias->printSize();
}

Sigmod::Sigmod(std::string name)
{
    m_name = name;
}

Matrix Sigmod::forward(Matrix &input) {
    return input.activation(utils::sigmod);
}

Matrix Sigmod::backward(Matrix &gardient){
    return gardient.derivation(utils::deltaSigmod);
}

ReLU::ReLU(std::string name)
{
    m_name = name;
}

Matrix ReLU::forward(Matrix &input) {
    activated = input.save_by_zero();
    return activated;
}

Matrix ReLU::backward(Matrix &gradient) {
    Matrix relu_activated = activated.save_by_zero();
    return  Matrix::hadamard(gradient,relu_activated);
}

}
