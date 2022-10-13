#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <vector>
#include "rcsc/layer.h"
namespace rcsc {
class Sequential
{
public:
    Sequential();
    std::vector<Layer *> graphs;
    std::vector<std::vector<Matrix *>> params;
    std::vector<std::vector<Matrix *>> grads;
    std::vector<std::vector<Matrix *>> veloc;
    int add(Layer *layer);
    Matrix forward(Matrix &input);
    int backward(Matrix &grad);
public:
    void print();
};

}

#endif // SEQUENTIAL_H
