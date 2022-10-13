#include "rcsc/optimizer.h"
#include <iostream>
namespace rcsc {

void SGD::clear_grad()
{
    for(int i = 0; i < m_grads.size(); i++)
    {
        m_grads[i][0]->clear();
        m_grads[i][1]->clear();
    }
}

Matrix SGD::get_decay(Matrix &grad)
{
    if (m_decay_type == "l1")
    {
        return grad ;
    }
    return grad * m_weight_decay;
}

SGD::SGD(std::vector<std::vector<Matrix *>> &params,
         std::vector<std::vector<Matrix *>> &grad,
         std::vector<std::vector<Matrix *>> &veloc,
         double learning_rete,
         double momentum,
         double weight_decay,
         std::string decay_type)
    :m_params(params),
    m_grads(grad),
    m_velocity(veloc),
    m_learning_rate(learning_rete),
    m_momentum(momentum),
    m_weight_decay(weight_decay)
{
    m_decay_type = decay_type;
}

void SGD::step()
{
    for (int i = 0; i < m_velocity.size(); i++)
    {
        for (int j = 0; j < 2; j++)
        {
           Matrix decay = get_decay(*m_grads[i][j]);
           Matrix *v = m_velocity[i][j];
           auto &grad = *m_grads[i][j];
           *v = grad + (*v) * m_momentum - decay;
           *m_params[i][j] = *m_params[i][j] - (*v) * m_learning_rate;
        }
    }
}

MSE::MSE(std::string reduction)
    :m_reduction(reduction){}

double MSE::get_loss(Matrix &pred,Matrix &target)
{
    double loss = 0.0;
    m_grad = pred - target;
    if (m_reduction == "mean")
    {
        loss = 0.5 * (pred - target).Square(2).mean();
    }
    else
    {
        loss = 0.5 * (pred - target).Square(2).sum();
    }
    return loss;
}

Matrix MSE::get_grad()
{
    return m_grad;
}
}

