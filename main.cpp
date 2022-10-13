#include <iostream>
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
    std::vector<Matrix> train_x;
    std::vector<Matrix> train_y;
    for(double i=1;i<=200;i++)
    {
        //double a = utils::randNum();
        double a = 1 + utils::randNum();
        std::vector<std::vector<double>> x;
        x.push_back(std::vector<double>());
        x[0].push_back(a);
        train_x.push_back(Matrix(x));
        x.clear();
        std::vector<std::vector<double>> y;
        y.push_back(std::vector<double>());
        y[0].push_back(a * a );
        train_y.push_back(Matrix(y));
        y.clear();
    }

    std::vector<std::vector<double>> x = {{1.2}};
    Matrix test_x = Matrix(x);
    std::vector<std::vector<double>> x1 = {{1.5}};
    Matrix test_x1 = Matrix(x1);
    
    Sequential Seq;
    Seq.add(new Linear("Linear1", 1, 16));
    //Seq.add(new Sigmod("Sigmod1"));
    Seq.add(new ReLU("RELU1"));
    Seq.add(new Linear("Linear2", 16, 64));
    //Seq.add(new Sigmod("Sigmod2"));
    Seq.add(new ReLU("RELU2"));
    Seq.add(new Linear("Linear3", 64, 1));
    //Seq.print();
    SGD *opt = new SGD(Seq.params,Seq.grads,Seq.veloc,0.001,0.9,0.1,"l2");
    MSE mse("mean");
    

    for (int epoch = 0; epoch < 30; epoch++)
    {
        for(int i = 0; i < 200; i++)
        {
            Matrix pred = Seq.forward(train_x[i]);
            double loss = mse.get_loss(pred,train_y[i]);
            Matrix grad = mse.get_grad();
            Seq.backward(grad);
            opt->step();
            opt->clear_grad();
            std::cout << "epoch : " << epoch << " step : " << i << " loss : " << loss << std::endl;
        }
    }
    
    Seq.forward(test_x).print();
    Seq.forward(test_x1).print();
    return 0;
}
