#ifndef NETWORK_H
#define NETWORK_H
#include"utils/matrix.h"
using namespace utils;
namespace rcsc {

class network
{
private:
    // Input Layer
    Matrix *input_value;
    Matrix *input_weight;

    // Hidden Layer
    std::vector<Matrix*> hidden_weight;
    std::vector<Matrix*> hidden_value;
    std::vector<Matrix*> hidden_value_a;
    std::vector<Matrix*> hidden_bias;

    // Output Layer
    Matrix *output_value;
    Matrix *output_bias;
    Matrix *output_value_a;

    // train data
    Matrix *data_input;
    Matrix *data_label;

    double learning_rate = 0.001;
public:
    network();
    ~network()
    {

    }
    void addInputLayer(int n);
    void addOutputLayer(int n);
    void addHiddenLayer(std::vector<int> hid_mat);
    void forwardPropagation();
    void backPropagation();
    void init();
    void initMatCalCore();
    void printNetworks();
    double cacLoss(Matrix &out,Matrix &label);
    void fit(std::vector<Matrix> &data_x,
             std::vector<Matrix> &data_y,
             double learningRate,
             int batchSzie,
             int epochs);
    void predict(Matrix &input);


};
}



#endif // NETWORK_H
