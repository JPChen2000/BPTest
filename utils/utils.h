#ifndef UTILS_H
#define UTILS_H
#include<vector>
#include<random>
#include<time.h>
#include"utils/matrix.h"

namespace utils {

inline double sigmod(double x){
    return 1.0 / (1 + exp(-x));
}

inline double relu(double x) {
    return x > 0 ? x:0;
}

inline double deltaRelu(double x)
{
    return x > 0? 1 : 0;
}

inline double deltaSigmod(double x)
{
    return x * (1-x);
}


static double randNum()
{

    std::random_device rd;  //如果可用的话，从一个随机数发生器上获得一个真正的随机数
    static std::mt19937 gen(rd()); //gen是一个使用rd()作种子初始化的标准梅森旋转算法的随机数发生器
    static std::uniform_int_distribution<> distrib(-10000,10000);
    return (double) distrib(gen)/10000;
//    std::default_random_engine rng(time(NULL));
//    std::uniform_real_distribution<double> distribution(-1, 1);
//    return distribution(rng);
}


static std::vector<double> randNums(int n)
{
    std::cout << "use randNums\n";
    // rand the random seed to get the different random result in each random
    std::random_device rd;  //如果可用的话，从一个随机数发生器上获得一个真正的随机数
    static std::mt19937 gen(rd()); //gen是一个使用rd()作种子初始化的标准梅森旋转算法的随机数发生器
    static std::uniform_int_distribution<> distribution(-10000,10000);
    std::vector<double > ret;
    for(int i = 0; i< n;i++)
    {
        ret.push_back((double)distribution(gen)/10000);
    }
    return ret;
}

}

#endif // UTILS_H
