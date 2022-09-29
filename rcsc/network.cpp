 #include "rcsc/network.h"
#include "utils/matrix.h"
#include "utils/utils.h"
#include<iostream>
using namespace utils;
namespace rcsc {
network::network()
{

}

void network::addInputLayer(int n)
{
    input_value =new Matrix(n);
}

void network::addOutputLayer(int n)
{
    hidden_weight.push_back(new Matrix(n,hidden_value[hidden_value.size()-1]->rows()));
    output_value = new Matrix(n);
    output_value_a = new Matrix(n);
    output_bias = new Matrix(n,1,true);
    data_label = new Matrix(n);
}

void network::addHiddenLayer(std::vector<int> hid_mat)
{
    for(size_t i=0;i<hid_mat.size();i++)
    {
        if( i==0 )
            input_weight = new Matrix(hid_mat[i],input_value->rows());
        else
            hidden_weight.push_back(new Matrix(hid_mat[i],hid_mat[i-1]));
        hidden_value.push_back(new Matrix(hid_mat[i]));
        hidden_value_a.push_back(new Matrix(hid_mat[i]));
        hidden_bias.push_back(new Matrix(hid_mat[i],1,true));
    }
}

void network::init()
{
    this->initMatCalCore();
}

void network::initMatCalCore()
{

}


void network::forwardPropagation()
{
    size_t i=0;

    for(;i<hidden_value.size();i++)
    {
        if(i==0) // 计算输入层
        {
            //*hidden_value[i] = Matrix::times(*input_weight,*input_value);
            *hidden_value[i] = Matrix::times(*input_weight,*input_value) + *hidden_bias[i];
            *hidden_value_a[i] = hidden_value[i]->activation(sigmod);
        }
        else //计算隐藏层
        {
            *hidden_value[i] = Matrix::times(*hidden_weight[i-1],*hidden_value_a[i-1]) + *hidden_bias[i];
            //*hidden_value[i] = Matrix::times(*hidden_weight[i-1],*hidden_value_a[i-1]);
            *hidden_value_a[i] = hidden_value[i]->activation(sigmod);
        }
    }
    // 计算输出层
    *output_value = Matrix::times(*hidden_weight[i-1],*hidden_value_a[i-1]) + *output_bias;
    //*output_value = Matrix::times(*hidden_weight[i-1],*hidden_value_a[i-1]);
    *output_value_a = output_value->activation(sigmod);
}

void network::backPropagation()
{
    Matrix back_mat;
    // 更新隐藏层权重
    for(int i=hidden_weight.size()-1;i>=0;i--)
    {
        if(i == (int)hidden_weight.size()-1) // 输出层前一层
        {
            back_mat = (*output_value_a - *data_label);
            back_mat = Matrix::hadamard(back_mat.rotate(),
                                        output_value_a->derivation(utils::deltaSigmod).rotate());
            *hidden_weight[i] += Matrix::times(*hidden_value[i],back_mat,true) * learning_rate;
            *output_bias += back_mat.rotate() * learning_rate;
//            back_mat = Matrix::times(back_mat,*hidden_weight[i]);
        }
        else
        {
            back_mat = Matrix::hadamard(hidden_value_a[i+1]->derivation(utils::deltaSigmod).rotate(),
                    Matrix::times(back_mat,*hidden_weight[i+1]));
            *hidden_weight[i] += Matrix::times(*hidden_value[i],back_mat,true) * learning_rate;
            *hidden_bias[i+1] += back_mat.rotate() * learning_rate ;
//            back_mat = Matrix::times(back_mat,*hidden_weight[i]);
        }
    }
    // 更新输入层权重
    back_mat = Matrix::hadamard(hidden_value_a[0]->derivation(utils::deltaSigmod).rotate(),
            Matrix::times(back_mat,*hidden_weight[0]));
    *input_weight += Matrix::times(*input_value,back_mat,true) * learning_rate;
    *hidden_bias[0] += back_mat.rotate() * learning_rate;
}

void network::fit(std::vector<Matrix> &data_x,
                  std::vector<Matrix> &data_y,
                  double learningRate,
                  int batchSzie,
                  int epochs)
{
    learning_rate = 1;
    int len = data_x.size();
    double loss = 1.0;
    for(int ep=0;ep<epochs;ep++)
    {
        for(int step=0;step<len/batchSzie;step++ )
        {
            for(int i=0;i<batchSzie;i++)
            {
                *input_value = data_x[step*batchSzie + i];
                *data_label = data_y[step*batchSzie + i];

                forwardPropagation();
//                input_value->print();
//                output_value_a->print();
                loss = cacLoss(*output_value_a,*data_label);
                //std::cout<<"predict:"<<output_value_a->mat[0][0]<<std::endl;
                std::cout<<"EPOCHS :"<<ep+1<<" STEP :"<<step+1<<"/"<<len/batchSzie<<" Loss :"<<loss
                        <<" Input:"<<input_value->mat[0][0]<<" Predict:"<<output_value_a->mat[0][0]<<std::endl;
                //printNetworks();
                backPropagation();

//                if(loss < 0.0001 && ep > 180)
//                    return;
            }

        }
    }
//    for(int i=0;i<2;i++)
//    {
//        *input_value = Matrix(data_x[0]);
//        data_label = new Matrix(data_y[0]);
//        forwardPropagation();
//        printNetworks();
//        double loss = cacLoss(*data_label,*output_value);
//        backPropagation();
//        printNetworks();
//        std::cout<<"loss :"<<loss<<std::endl;
//    }
}

double network::cacLoss(Matrix &out,Matrix &label)
{
    double avgLoss = 0;
    for(int i=0;i<out.mat.size();i++)
    {
        avgLoss += pow((out.mat[i][0]-label.mat[i][0]),2);
    }
    avgLoss = 0.5 * avgLoss / out.mat.size();
    return avgLoss;
}

void network::predict(Matrix &input)
{
    *input_value = input;
//    output_value_a->print();
//    input_value->print();
    forwardPropagation();
    std::cout<<"predict:"<<output_value_a->mat[0][0]<<std::endl;
}

void network::printNetworks()
{
    std::cout<<"Input Layer:"<<std::endl;
    input_value->printSize();
    input_value->print();
    std::cout<<"weight"<<std::endl;
    input_weight->printSize();
    input_weight->print();

    std::cout<<std::endl;   

    std::cout<<"Hidden Layer:"<<std::endl;
    int i=0;
    for(auto &layer:hidden_value)
    {
        std::cout<<"Layer "<<i++<<std::endl;
        layer->printSize();
        layer->print();
        std::cout<<std::endl;
    }
    std::cout<<"weight"<<std::endl;;
    i=0;
    for(auto &layer:hidden_weight)
    {
        std::cout<<"Layer "<<i++<<std::endl;
        layer->printSize();
        layer->print();
        std::cout<<std::endl;
    }

    std::cout<<"bias"<<std::endl;
    i=0;
    for(auto &layer:hidden_bias)
    {
        std::cout<<"Layer "<<i++<<std::endl;
        layer->printSize();
        layer->print();
        std::cout<<std::endl;
    }

    std::cout<<"Output Layer:"<<std::endl;
    output_value->printSize();
    output_value->print();
    std::cout<<"bias"<<std::endl;
    output_bias->printSize();
    output_bias->print();
}


}
