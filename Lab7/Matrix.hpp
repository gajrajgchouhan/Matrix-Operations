#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <set>
#include <complex>
using namespace std;

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class Matrix
{
private:
    size_t mRows;
    size_t mCols;
    std::vector<T> mData;

    void validateIndexes(size_t row, size_t col) const
    {
        if (row < 0 or row >= mRows)
            throw invalid_argument(
                "Invalid row index (" + to_string(row) + "): should be between 0 and " + to_string(mRows - 1));
        if (col < 0 or col >= mCols)
            throw invalid_argument(
                "Invalid column index (" + to_string(col) + "): should be between 0 and " + to_string(mCols - 1));
    }

public:
    size_t nCols() const { return mCols; }

    size_t nRows() const { return mRows; }

    Matrix()
    {
        mRows = mCols = 0;
    }

    Matrix(size_t n) : mRows(n), mCols(n), mData(n * n) {}

    Matrix(size_t rows, size_t cols) : mRows(rows), mCols(cols), mData(rows * cols) {}

    Matrix(size_t rows, size_t cols, const vector<T> &data) : mRows(rows), mCols(cols)
    {
        if (data.size() != rows * cols)
            throw invalid_argument("Matrix dimension incompatible with its initializing vector.");
        mData = data;
    }

    size_t size() const
    {
        return mRows * mCols;
    }

    friend Matrix operator+(const Matrix &m, double value)
    {
        Matrix result(m.mRows, m.mCols);

        for (size_t i = 0; i < m.mRows; i++)
        {
            for (size_t j = 0; j < m.mCols; j++)
            {
                result(i, j) = value + m(i, j);
            }
        }

        return result;
    }

    friend Matrix operator+(double value, const Matrix &m)
    {
        return m + value;
    }

    friend Matrix operator-(const Matrix &m, double value)
    {
        return m + (-value);
    }

    //! Scalar subtraction
    friend Matrix operator-(double value, const Matrix &m)
    {
        return m - value;
    }

    friend Matrix operator*(const Matrix &m, double value)
    {
        Matrix result(m.mRows, m.mCols);

        for (size_t i = 0; i < m.mRows; i++)
        {
            for (size_t j = 0; j < m.mCols; j++)
            {
                result(i, j) = value * m(i, j);
            }
        }

        return result;
    }

    friend Matrix operator*(double value, const Matrix &m)
    {
        return m * value;
    }

    friend Matrix operator/(const Matrix &m, double value)
    {
        Matrix result(m.mRows, m.mCols);

        for (size_t i = 0; i < m.mRows; i++)
        {
            for (size_t j = 0; j < m.mCols; j++)
            {
                result(i, j) = m(i, j) / value;
            }
        }

        return result;
    }

    friend Matrix operator/(double value, const Matrix &m)
    {
        Matrix result(m.mRows, m.mCols);

        for (size_t i = 0; i < m.mRows; i++)
        {
            for (size_t j = 0; j < m.mCols; j++)
            {
                result(i, j) = value / m(i, j);
            }
        }

        return result;
    }

    Matrix operator+=(double value)
    {
        for (int i = 0; i < mData.size(); i++)
            mData[i] += value;
        return *this;
    }

    Matrix operator-=(double value)
    {
        for (int i = 0; i < mData.size(); i++)
            mData[i] -= value;
        return *this;
    }

    Matrix operator*=(double value)
    {
        for (int i = 0; i < mData.size(); i++)
            mData[i] *= value;
        return *this;
    }

    Matrix operator/=(double value)
    {
        for (int i = 0; i < mData.size(); i++)
            mData[i] /= value;
        return *this;
    }

    Matrix operator+(const Matrix &b)
    {
        if (mRows != b.mRows || mCols != b.mCols)
            throw invalid_argument("Cannot add these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));

        Matrix result(mRows, mCols);

        for (size_t i = 0; i < mRows; i++)
        {
            for (size_t j = 0; j < mCols; j++)
            {
                result(i, j) = operator()(i, j) + b(i, j);
            }
        }

        return result;
    }

    Matrix operator-(const Matrix &b)
    {
        if (mRows != b.mRows || mCols != b.mCols)
            throw invalid_argument(
                "Cannot subtract these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));

        Matrix result(mRows, mCols);

        for (size_t i = 0; i < mRows; i++)
        {
            for (size_t j = 0; j < mCols; j++)
            {
                result(i, j) = operator()(i, j) - b(i, j);
            }
        }

        return result;
    }

    Matrix operator/(const Matrix &b)
    {
        if (mRows != b.mRows || mCols != b.mCols)
            throw invalid_argument(
                "Cannot divide these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));

        Matrix result(mRows, mCols);

        for (size_t i = 0; i < mRows; i++)
        {
            for (size_t j = 0; j < mCols; j++)
            {
                result(i, j) = operator()(i, j) / b(i, j);
            }
        }

        return result;
    }

    Matrix operator*(const Matrix &b) const
    {
        if (mCols != b.mRows)
            throw invalid_argument(
                "Cannot multiply these matrices: L = " + to_string(this->mRows) + "x" +
                to_string(this->mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));

        Matrix result = zeros(mRows, b.mCols);

        for (size_t i = 0; i < result.mRows; i++)
        {
            for (size_t k = 0; k < mCols; k++)
            {
                double tmp = operator()(i, k);
                for (size_t j = 0; j < result.mCols; j++)
                {
                    result(i, j) += tmp * b(k, j);
                }
            }
        }

        return result;
    }

    Matrix &operator+=(const Matrix &other)
    {
        if (mRows != other.mRows || mCols != other.mCols)
            throw invalid_argument("Cannot add these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(other.mRows) + "x" + to_string(other.mCols));
        for (size_t i = 0; i < other.mRows; i++)
        {
            for (size_t j = 0; j < other.mCols; j++)
            {
                operator()(i, j) += other(i, j);
            }
        }

        return *this;
    }

    Matrix &operator-=(const Matrix &other)
    {
        if (mRows != other.mRows || mCols != other.mCols)
            throw invalid_argument(
                "Cannot subtract these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(other.mRows) + "x" + to_string(other.mCols));

        for (size_t i = 0; i < other.mRows; i++)
        {
            for (size_t j = 0; j < other.mCols; j++)
            {
                operator()(i, j) -= other(i, j);
            }
        }

        return *this;
    }

    Matrix &operator*=(const Matrix &other)
    {
        if (mCols != other.mRows)
            throw invalid_argument(
                "Cannot multiply these matrices: L " + to_string(mRows) + "x" +
                to_string(mCols) + ", R " + to_string(other.mRows) + "x" + to_string(other.mCols));

        Matrix result(mRows, other.mCols);

        for (size_t i = 0; i < result.mRows; i++)
        {
            for (size_t j = 0; j < result.mCols; j++)
            {
                result(i, j) = 0;
                for (size_t ii = 0; ii < mCols; ii++)
                    result(i, j) += operator()(i, ii) * other(ii, j);
            }
        }

        mRows = result.mRows;
        mCols = result.mCols;
        mData = result.mData;
        return *this;
    }

    Matrix<int> operator==(const T &value)
    {
        Matrix<int> result(mRows, mCols);

        for (size_t i = 0; i < mRows; i++)
        {
            for (size_t j = 0; j < mCols; j++)
            {
                result(i, j) = operator()(i, j) == value;
            }
        }

        return result;
    }

    bool operator==(const Matrix &other)
    {
        if (mData.size() != other.mData.size() || mRows != other.mRows || mCols != other.mCols)
            return false;

        for (int k = 0; k < mData.size(); k++)
        {
            if (mData[k] != other.mData[k])
                return false;
        }

        return true;
    }

    Matrix operator-()
    {
        Matrix result(this->mRows, this->mCols);

        for (size_t i = 0; i < mCols; i++)
        {
            for (size_t j = 0; j < mRows; j++)
            {
                result(i, j) = -operator()(i, j);
            }
        }

        return result;
    }

    T &operator()(size_t i, size_t j)
    {
        validateIndexes(i, j);
        return mData[i * mCols + j];
    }

    T operator()(size_t i, size_t j) const
    {
        validateIndexes(i, j);
        return mData[i * mCols + j];
    }

    static Matrix fill(size_t rows, size_t cols, double value)
    {
        Matrix result(rows, cols, vector<T>(rows * cols, value));
        return result;
    }

    static Matrix diagonal(size_t size, double value)
    {
        Matrix result = zeros(size, size);
        for (size_t i = 0; i < size; i++)
            result(i, i) = value;

        return result;
    }

    bool isSquare() const
    {
        return mCols == mRows;
    }

    Matrix diagonal()
    {
        if (!isSquare())
        {
            throw runtime_error("Can't get the diagonal, not a square matrix");
        }

        Matrix<T> result(mRows, 1);

        for (size_t i = 0; i < mRows; i++)
            result(i, 0) = operator()(i, i);

        return result;
    }

    static Matrix identity(size_t size)
    {
        return diagonal(size, 1);
    }

    static Matrix ones(size_t rows, size_t cols)
    {
        return fill(rows, cols, 1);
    }

    static Matrix zeros(size_t rows, size_t cols)
    {
        return fill(rows, cols, 0);
    }

    Matrix transpose() const
    {
        Matrix result(mCols, mRows);

        for (size_t i = 0; i < mRows; i++)
        {
            for (size_t j = 0; j < mCols; j++)
            {
                result(j, i) = operator()(i, j);
            }
        }

        return result;
    }

    Matrix getColumn(size_t index)
    {
        if (index >= mCols)
            throw invalid_argument("Column index out of bounds");

        Matrix result(mRows, 1);
        for (size_t i = 0; i < mRows; i++)
            result(i, 0) = operator()(i, index);

        return result;
    }

    Matrix getRow(size_t index)
    {
        if (index >= mRows)
            throw invalid_argument("Row index out of bounds");

        Matrix result(mCols, 1);
        for (size_t i = 0; i < mCols; i++)
            result(i, 0) = operator()(index, i);

        return result;
    }

    friend ostream &operator<<(ostream &os, const Matrix &matrix)
    {
        const int numWidth = 13;
        char fill = ' ';

        for (int i = 0; i < matrix.mRows; i++)
        {
            for (int j = 0; j < matrix.mCols; j++)
            {
                // the trick to print a table-like structure was stolen from here
                // https://stackoverflow.com/a/14796892
                os << left << setw(numWidth) << setfill(fill) << to_string(matrix(i, j));
            }
            os << endl;
        }

        return os;
    }

    Matrix asDiagonal()
    {
        if (mRows != 1 and mCols != 1)
            throw runtime_error("Can't diagonalize, not a vector");

        size_t dimension = mCols > 1 ? mCols : mRows;

        Matrix result = zeros(dimension, dimension);

        for (size_t i = 0; i < dimension; i++)
        {
            result(i, i) = mCols > 1 ? operator()(0, i) : operator()(i, 0);
        }
        return result;
    }

    Matrix copy()
    {
        Matrix result(mRows, mCols);
        result.mData = mData;
        return result;
    }

    T sum() const
    {
        T sum_of_elems = 0;
        for (T n : mData)
            sum_of_elems += n;

        return sum_of_elems;
    }

    T min() const
    {
        return *min_element(mData.begin(), mData.end());
    }

    T max() const
    {
        return *max_element(mData.begin(), mData.end());
    }

    float norm()
    {
        float norm = 0;
        for (float cell : mData)
        {
            norm += cell * cell;
        }
        return sqrt(norm);
    }

    operator int()
    {
        if (mRows == 1 && mCols == 1)
        {
            return (int)mData[0];
        }
        else
        {
            throw invalid_argument("Matrix has more than one element");
        }
    }

    operator float()
    {
        if (mRows == 1 && mCols == 1)
        {
            return (float)mData[0];
        }
        else
        {
            throw invalid_argument("Matrix has more than one element");
        }
    }

    operator double()
    {
        if (mRows == 1 && mCols == 1)
        {
            return (double)mData[0];
        }
        else
        {
            throw invalid_argument("Matrix has more than one element");
        }
    }

    typename vector<T>::iterator begin()
    {
        return mData.begin();
    }

    typename vector<T>::iterator end()
    {
        return mData.end();
    }

    void setRow(size_t index, Matrix<T> row)
    {
        if (mRows < index)
            throw invalid_argument("Invalid row index, matrix is not that large");
        if (mCols != row.mCols)
            throw invalid_argument("Incompatible number of columns");
        if (row.mRows > 1)
            throw invalid_argument("Row matrix contains more than one row");

        for (size_t col = 0; col < mCols; col++)
            operator()(index, col) = row(0, col);
    }

    void setColumn(size_t index, Matrix<T> column)
    {
        if (mCols < index)
            throw invalid_argument("Invalid row column, matrix is not that large");
        if (mRows != column.mRows)
            throw invalid_argument("Incompatible number of rows");
        if (column.mCols > 1)
            throw invalid_argument("Column matrix contains more than one column");

        for (size_t row = 0; row < mCols; row++)
            operator()(row, index) = column(row, 0);
    }

    static pair<Matrix, Matrix> eigsort(Matrix eigenvalues, Matrix eigenvectors)
    {
        Matrix eigval(eigenvalues.mRows, eigenvalues.mCols, eigenvalues.mData);
        Matrix eigvec(eigenvectors.mRows, eigenvectors.mCols, eigenvectors.mData);

        vector<size_t> newOrder;
        for (size_t i = 0; i < eigenvalues.nRows(); i++)
        {
            int position = 0;
            for (int j = 0; j < newOrder.size(); j++)
                if (eigenvalues(i, 0) < eigenvalues(newOrder[j], 0))
                    position++;
            newOrder.insert(newOrder.begin() + position, i);
        }

        for (size_t i = 0; i < newOrder.size(); i++)
        {
            eigval(i, 0) = eigenvalues(newOrder[i], 0);

            for (int j = 0; j < eigenvectors.nRows(); j++)
            {
                eigvec(static_cast<size_t>(j), i) = eigenvectors(j, newOrder[i]);
            }
        }

        return make_pair(eigval, eigvec);
    }
};

typedef Matrix<double> MatrixD;
typedef Matrix<float> MatrixF;
typedef Matrix<int> MatrixI;

#endif //MATRIX_HPP