#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include"utils/matrix.h"

using namespace utils;
namespace rcsc {

class Layer{
public:
    Layer() {}
    Layer(std::string name):m_name(name) {}
    virtual Matrix forward(Matrix &input) {}
    virtual Matrix backward(Matrix &gradient) {}
    std::vector<Matrix *> get_params();
    std::vector<Matrix *> get_grads();
    std::vector<Matrix *> get_veloc();
    void clear_grad();
    bool is_activate_layer();
    virtual void print();
protected:
    Matrix *weight;
    Matrix *weight_grad;
    Matrix *bias;
    Matrix *bias_grad;
    Matrix *veloc_weight_grad;
    Matrix *veloc_bias_grad;
    Matrix input;
    std::string m_name;
};

class Linear : public Layer{
public:
    Linear() = delete;
    Linear(std::string name,int input_size,int output_size);

    Matrix forward(Matrix &input);
    Matrix backward(Matrix &gradient);
    void print();
};

class ReLU : public Layer{
public:
    ReLU(std::string name);
    Matrix forward(Matrix &input);
    Matrix backward(Matrix &gradient);
protected:
    Matrix activated;
};

class Sigmod : public Layer {
public:
    Sigmod(std::string name);
    Matrix forward(Matrix &input);
    Matrix backward(Matrix &gardient);
};

}


#endif // LAYER_H
