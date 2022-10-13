#ifndef MATRIX_H
#define MATRIX_H
#include<vector>
#include<iostream>
namespace utils {

class Matrix
{
private:
    int col,row;
public:
    std::vector<std::vector<double>> mat;
    Matrix();
    Matrix(int n,int m=1,bool normalize = false);
    Matrix(std::vector<std::vector<double>> &Mat);
    ~Matrix(){

    }
    
    void init();
    void print();
    void printSize();
    void clear();
    int rows();
    int cols();
    double sum();
    double mean();
    Matrix sum_axis_0();
    Matrix activation(double (*func)(double));
    Matrix derivation(double (*func)(double));
    void normalization();
    Matrix rotate();
    Matrix Square(double x);
    Matrix fabs();
    Matrix zeros_like();
    Matrix save_by_zero();
    
    Matrix &operator = (const Matrix & M);
    Matrix operator + (const Matrix & M);
    Matrix operator - (const Matrix & M);
    Matrix &operator += (const Matrix & M);
    Matrix &operator -= (const Matrix & M);
    Matrix operator * (const double & l);
    Matrix &operator < (const double &l);
    Matrix &operator > (const double &l);

    static Matrix hadamard(Matrix a,Matrix b);
    static Matrix times( Matrix &a,Matrix &b,bool trans = false);
    static double tr(Matrix &M);
    static Matrix delta(Matrix &a,Matrix &b);
    static Matrix & derivation(Matrix &a,Matrix &b);
};
}

#endif // MATRIX_H
