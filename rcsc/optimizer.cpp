//#include "optimizer.h"
//namespace rcsc {
//Optimizer::Optimizer()
//{

//}

//Optimizer::Optimizer(std::vector<std::vector<Matrix>> &params,std::vector<std::vector<Matrix>> grad, double learning_rete, double weight_decay,std::string decay_type)
//    :m_params(params),
//     m_grads(grad),
//     m_learning_rate(learning_rete),
//     m_weight_decay(weight_decay),
//     m_decay_type(decay_type) {}

//void Optimizer::step() {}

//void Optimizer::clear_grad()
//{
//    for(int i = 0; i < m_grads.size(); i++)
//    {
//        m_grads[i][0].clear();
//        m_grads[i][1].clear();
//    }
//}

//Matrix Optimizer::get_decay(Matrix &grad)
//{
//    if (m_decay_type == "l1")
//    {
//        return m_weight_decay;
//    }
//    else if (m_decay_type == "l2")
//    {
//        return grad * m_weight_decay;
//    }
//    return grad;
//}

//SGD::SGD(std::vector<std::vector<Matrix>> &params,
//         std::vector<std::vector<Matrix>> &grad,
//         double learning_rete,
//         double momentum,
//         double weight_decay,
//         std::string decay_type)
//    :m_params(params),
//    m_grads(grad),
//    m_learning_rate(learning_rete),
//    m_momentum(momentum),
//    m_weight_decay(weight_decay),
//    m_decay_type(decay_type)
//{
//    m_velocity.resize(m_params.size());
//    for(int i = 0; i < m_velocity.size(); i++)
//    {
//        m_velocity[i].resize(2);
//        m_velocity[i].push_back(m_grads[i][0].zeros_like());
//        m_velocity[i].push_back(m_grads[i][1].zeros_like());
//    }
//}

//void SGD::step()
//{
//    for (int i = 0; i < m_velocity.size(); i++)
//    {
//        for (int j = 0; j < 1; j++)
//        {
//           Matrix decay = get_decay(m_grads[i][j]);
//           Matrix &v = m_velocity[i][j];
//           v = v * m_momentum + m_grads[i][j] + decay;
//           m_params[i][j] = m_params[i][j] - v * m_learning_rate;
//        }
//    }
//}

//MSE::MSE(std::string reduction)
//    :m_reduction(reduction){}

//double MSE::get_loss(Matrix &pred,Matrix &target)
//{
//    double loss = 0.0;
//    if (m_reduction == "mean")
//    {
//        loss = 0.5 * (pred - target).Square(2).mean();
//    }
//    else
//    {
//        loss = 0.5 * (pred - target).Square(2).sum();
//    }
//    return loss;
//}

//}

