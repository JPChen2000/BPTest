#include <iostream>
#include <rcsc/network.h>
#include <utils/matrix.h>
#include <utils/utils.h>
#include <rcsc/layer.h>
#include <rcsc/optimizer.h>
#include <rcsc/sequential.h>
#include <vector>
using namespace rcsc;

int main()
{

//    Matrix mat1(x1);
//    Matrix mat2(x2);
//    mat1.printSize();
//    mat2.printSize();
//    Matrix mat3 = Matrix::times(mat1,mat2,true);
//    mat3.printSize();
//    std::vector<std::vector<double>> t = {{0.1}};
//    Matrix input = Matrix(t);
//    network net;
//    net.addInputLayer(1);
//    net.addHiddenLayer({2,2});
//    net.addOutputLayer(1);
//    //net.printNetworks();
    std::vector<Matrix> train_x;
    std::vector<Matrix> train_y;
    for(double i=1;i<=200;i++)
    {
        double a = utils::randNum();
        std::vector<std::vector<double>> x;
        x.push_back(std::vector<double>());
        x[0].push_back(a);
        train_x.push_back(Matrix(x));
        x.clear();
        std::vector<std::vector<double>> y;
        y.push_back(std::vector<double>());
        y[0].push_back(a * a);
        train_y.push_back(Matrix(y));
        y.clear();
    }
//    net.predict(input);
//    net.fit(train_x,train_y,0.005,1,1);
//    net.predict(input);

    //net.printNetworks();

    //test
//    for(int i=0;i<10;i++)
//    {
//    net.forwardPropagation();
//    net.backPropagation();
//    }
    //net.printNetworks();

//    Sequential Seq;

//    Seq.add(new Linear(1,16));
//    Seq.add(new ReLU());
//    Seq.add(new Linear(16,64));
//    Seq.add(new ReLU());
//    Seq.add(new Linear(64,1));

//    SGD opt(Seq.params,Seq.grads,0.005);
//    MSE mse("mean");

//    for (int epoch = 0; epoch < 10; epoch++)
//    {
//        for(int i = 0; i < 200; i++)
//        {
//            Matrix pred = Seq.forward(train_x[i]);
//            double loss = mse.get_loss(pred,train_y[i]);
//            Matrix grad = mse.get_grad();
//            Seq.backward(grad);

//            opt.step();
//            opt.clear_grad();
//            std::cout << "epoch : " << epoch << " step : " << i << " loss : " << loss << std::endl;
//        }
//    }

    return 0;
}
