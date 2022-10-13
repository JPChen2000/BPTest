#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>
#include <list>
#include "rcsc/layer.h"

namespace rcsc {
class SGD
{
public:
    SGD() {}
    SGD(std::vector<std::vector<Matrix*>> &params,
        std::vector<std::vector<Matrix*>> &grad,
        std::vector<std::vector<Matrix*>> &veloc,
        double learning_rete,
        double momentum = 0.9,
        double weight_decay = 0.1,
        std::string decay_type = "l2");
    void step();
    void clear_grad();
    Matrix get_decay(Matrix &grad);
protected:
    std::vector<std::vector<Matrix *>> m_params;
    std::vector<std::vector<Matrix *>> m_grads;
    std::vector<std::vector<Matrix *>> m_velocity;
    double m_learning_rate;
    double m_momentum;
    double m_weight_decay;
    std::string m_decay_type;

};

class MSE
{
public:
    MSE(std::string reduction = "mean");
    double get_loss(Matrix &pred, Matrix &target);
    Matrix get_grad();
protected:
    std::string m_reduction;
    Matrix m_grad;
};
}

#endif // OPTIMIZER_H
