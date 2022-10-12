#include "rcsc/sequential.h"
namespace rcsc {
Sequential::Sequential() {}

int Sequential::add(Layer *layer)
{
    graphs.push_back(layer);
    if (layer->is_activate_layer())
        return 0;
    params.push_back(layer->get_params());
    grads.push_back(layer->get_grads());
    veloc.push_back(layer->get_veloc());
    return 0;
}

Matrix Sequential::forward(Matrix &input)
{
    Matrix layer_input = input;
    for (auto &layer:graphs)
    {   
        layer_input = layer->forward(layer_input);
    }
    return layer_input;
}

int Sequential::backward(Matrix &grad)
{
    Matrix layer_grad = grad;
    for (int i=graphs.size()-1; i >= 0; i--)
    {
        auto &layer = graphs[i];
        layer_grad = layer->backward(layer_grad);
    }
    return 0;
}

void Sequential::printWeights()
{
    for (auto &layer : graphs)
    {
        layer->print();
        std::cout << "is activate layer" << layer->is_activate_layer() << std::endl;
        if (!layer->is_activate_layer())
        {
            layer->printWeights();
        }    
    }
}

void Sequential::print()
{
    for (auto &layer : graphs)
    {
        layer->print();
    }
}

}

