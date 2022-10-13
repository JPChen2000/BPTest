#include<iostream>
#include<utils/matrix.h>
#include<utils/utils.h>
namespace utils {


Matrix::Matrix()
{
    row = 0;
    col = 0;
}

Matrix::Matrix(int n,int m,bool normalize)
{
    row = n;
    col = m;
    mat.resize(row);
    for(auto & r:mat)
        r.resize(col);
    if(  !normalize )
        init();
    else
        normalization();

}

Matrix::Matrix(std::vector<std::vector<double>> &Mat)
    :mat(Mat)
{
    row = Mat.size();
    col = Mat[0].size();
    mat.resize(row);
    for(auto & r:mat)
        r.resize(col);
}

Matrix Matrix::activation(double (*func)(double))
{
    Matrix tmp = *this;
    for(auto &ro:tmp.mat)
    {
        for(auto & x:ro)
            x = func(x);
    }
    return tmp;
}

Matrix Matrix::derivation(double (*func)(double))
{
    Matrix temp = *this;
    for(auto & ro:temp.mat)
        for(auto &x:ro)
            x = func(x);
    return temp;
}

void Matrix::init()
{

    for(auto &row:mat)
    {
        std::vector<double> tmp = randNums(col);
        row.assign(tmp.begin(),tmp.end());
    }
}

void Matrix::normalization()
{
    for(auto &row:mat)
    {
        row.assign(col,0.0f);
    }
}

void Matrix::clear()
{
    for(size_t i=0;i<mat.size();i++)
    {
        for(size_t j=0;j<mat[0].size();j++)
        {
           mat[i][j] = 0;
        }
    }
}

Matrix Matrix::sum_axis_0()
{
    Matrix tmp(1, col);
    tmp.clear();
    for (int i = 0; i < tmp.mat[0].size(); i++)
    {
        for (int j = 1; j < mat.size(); j++)
        {
            tmp.mat[0][i] += mat[j][i];
        }
    }
    return tmp;
}

double Matrix::sum()
{
    double sum = 0;
    for(size_t i=0;i<mat[0].size();i++)
    {
        for(size_t j=0;j<mat.size();j++)
        {
           sum += mat[i][j];
        }
    }
    return sum;
}

double Matrix::mean()
{
    double sum = 0;
    double size = 0;
    for(size_t i=0;i<mat[0].size();i++)
    {
        for(size_t j=0;j<mat.size();j++)
        {
           sum += mat[i][j];
           size ++;
        }
    }
    return sum / size;
}

Matrix Matrix::zeros_like()
{
    Matrix ret(mat.size(), mat[0].size());
    ret.mat = mat;
    ret.clear();
    return ret;
}

Matrix Matrix::rotate()
{
    std::vector<std::vector<double>> tmp;
    for(size_t i=0;i<mat[0].size();i++)
    {
        tmp.push_back(std::vector<double>());
        for(size_t j=0;j<mat.size();j++)
        {
            tmp[i].push_back(mat[j][i]);
        }
    }
    return Matrix(tmp);
}

Matrix Matrix::Square(double x)
{
    for(auto &ro:mat)
    {
        for(auto &it:ro)
        {
            it = pow(it,x);
        }
    }
    return *this;
}

Matrix Matrix::fabs()
{
    for(auto &ro:mat)
    {
        for(auto &it:ro)
        {
            it = std::fabs(it);
        }
    }
    return *this;
}

Matrix Matrix::save_by_zero()
{
    Matrix tmp = *this;
    for(size_t i=0;i<mat.size();i++)
    {
        for(size_t j=0;j<mat[0].size();j++)
        {
            tmp.mat[i][j] = tmp.mat[i][j] > 0 ? tmp.mat[i][j] : 0;
        }
    }
    return tmp;

}


Matrix &Matrix::operator = (const Matrix & M)
{
    mat = M.mat;
    this->row = mat.size();
    this->col = mat[0].size();
    return *this;
}

Matrix Matrix::operator + (const Matrix &M)
{
    Matrix tmp = *this;
    for(size_t i=0;i<mat.size();i++)
    {
        for(size_t j=0;j<mat[0].size();j++)
        {
            tmp.mat[i][j] += M.mat[i][j];
        }
    }
    return tmp;
}

Matrix Matrix::operator - ( const Matrix &M)
{
    Matrix tmp = *this;
    for(size_t i=0;i<mat.size();i++)
    {
        for(size_t j=0;j<mat[0].size();j++)
        {
            tmp.mat[i][j] -= M.mat[i][j];
        }
    }
    return tmp;
}

Matrix &Matrix::operator+=(const Matrix &M)
{
    *this = *this + M;
    return *this;
}

Matrix & Matrix::operator-=(const Matrix &M)
{
    *this = *this - M;
    return *this;
}


Matrix Matrix::operator*(const double &l)
{
    Matrix tmp = *this;
    for(size_t i=0;i<mat.size();i++)
    {
        for(size_t j=0;j<mat[0].size();j++)
        {
            tmp.mat[i][j] *= l;
        }
    }
    return tmp;
}

Matrix & Matrix::operator<(const double &l)
{
    for(size_t i=0;i<mat.size();i++)
    {
        for(size_t j=0;j<mat[0].size();j++)
        {
            if (mat[i][j] > l)
            {
                mat[i][j] = 0;
            }
        }
    }
    return *this;
}

Matrix & Matrix::operator>(const double &l)
{
    for(size_t i=0;i<mat.size();i++)
    {
        for(size_t j=0;j<mat[0].size();j++)
        {
            if (mat[i][j] < l)
            {
                mat[i][j] = 0;
            }
        }
    }
    return *this;
}

int Matrix::cols()
{
    return this->col;
}

int Matrix::rows()
{
    return this->row;
}

void Matrix::printSize()
{
    std::cout<<row<<"x"<<col;
}

void Matrix::print()
{
    for(auto &row:mat)
    {
        for(auto &idx:row)
            std::cout<<idx<<" ";
        std::cout<<std::endl;
    }
}

Matrix Matrix::hadamard(Matrix a, Matrix b)
{
    std::vector<std::vector<double>> temp;
    for(size_t i=0;i<a.mat.size();i++)
    {
        temp.push_back(std::vector<double>());
        for(size_t j=0;j<a.mat[0].size();j++)
        {
            temp[i].push_back(a.mat[i][j] * b.mat[i][j]);
        }
    }
    return Matrix(temp);
}

Matrix Matrix::times(Matrix &a,Matrix &b,bool trans)
{
    size_t n=a.mat.size();
    size_t m=a.mat[0].size();
    size_t s=b.mat[0].size();
    std::vector<std::vector<double>> temp;
    if (!trans)
    {
        for(size_t i=0;i< n;i++)
        {
            temp.push_back(std::vector<double>());
            for(size_t j=0;j<s;j++)
            {
                double t = 0;
                for(size_t k=0;k<m;k++)
                {
                    t+=a.mat[i][k] * b.mat[k][j];
                }
                temp[i].push_back(t);
            }
        }
    }
    else
    {
        for(size_t i=0;i<s;i++)
        {
            temp.push_back(std::vector<double>());
            for(size_t j=0;j<n;j++)
            {
                double t = 0;
                for(size_t k=0;k<m;k++)
                {
                    t+=a.mat[j][k] * b.mat[k][i];
                }
                temp[i].push_back(t);
            }
        }
    }
    return Matrix(temp);
}

double Matrix::tr(Matrix &M)
{
    double temp = 0;
    for(size_t i=0;i<M.mat.size();i++)
    {
        temp += M.mat[i][i];
    }
    return temp;
}


}
