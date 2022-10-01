#include <iostream>
#include <rcsc/network.h>
#include <utils/matrix.h>
#include <utils/utils.h>
#include <rcsc/layer.h>
#include <rcsc/optimizer.h>
#include <rcsc/sequential.h>
#include <vector>
using namespace rcsc;
using utils::Matrix;
int main()
{
//      std::vector<std::vector<double>> x1 = {{-1,2},{3,4},{5,6}};
//      std::vector<std::vector<double>> x2 = {{1},{0}};
//      std::vector<std::vector<double>> x3 = {{1},{1}};
//      Matrix mat1(x1);
//      Matrix mat2(x2);
//      mat1.printSize();
//      mat2.printSize();
//      Matrix mat3 = Matrix::times(mat1,mat2,true);
//      mat3.printSize();
//      mat3.print();
//      Matrix mat4 = mat1.zeros_like();
//      mat4.print();
//      Matrix mat5(x3);
//      Matrix mat6 = mat5 + mat2;
//      mat6.print();
//      mat1.rotate().print();
//      std::cout<<mat1.sum()<< " "<<mat1.mean();
//      mat1.save_by_zero().print();
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
      //  double a = utils::randNum() * 10;
        double a = 1;
        std::vector<std::vector<double>> x;
        x.push_back(std::vector<double>());
        x[0].push_back(a);
        train_x.push_back(Matrix(x));
        x.clear();
        std::vector<std::vector<double>> y;
        y.push_back(std::vector<double>());
        y[0].push_back(a * a + 0.01 * utils::randNum());
        train_y.push_back(Matrix(y));
        y.clear();
    }

    std::vector<std::vector<double>> x = {{1.5}};
    Matrix test_x = Matrix(x);

////    net.predict(input);
////    net.fit(train_x,train_y,0.005,1,1);
////    net.predict(input);
//
//    //net.printNetworks();
//
//    //test
////    for(int i=0;i<10;i++)
////    {
////    net.forwardPropagation();
////    net.backPropagation();
////    }
//    //net.printNetworks();
//
    Sequential Seq;

    Seq.add(new Linear("Linear1", 1, 16));
    Seq.add(new ReLU("RELU1"));
    Seq.add(new Linear("Linear2", 16, 64));
    Seq.add(new ReLU("RELU2"));
    Seq.add(new Linear("Linear3", 64, 1));
    Seq.print();
    SGD opt(Seq.params,Seq.grads,Seq.veloc,0.00001);
    MSE mse("mean");
    

    for (int epoch = 0; epoch < 10; epoch++)
    {
        for(int i = 0; i < 200; i++)
        {
            Matrix pred = Seq.forward(train_x[i]);
            double loss = mse.get_loss(pred,train_y[i]);
            Matrix grad = mse.get_grad();
            Seq.backward(grad);

            opt.step();
            opt.clear_grad();
            std::cout << "epoch : " << epoch << " step : " << i << " loss : " << loss << std::endl;
        }
    }
    
    
    Seq.forward(test_x).print();
    return 0;
}
