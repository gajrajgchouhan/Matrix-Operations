#ifndef SMATRIX_HPP
#define SMATRIX_HPP

#include <unordered_map>
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <set>
#include <complex>
using namespace std;

struct HASH
{
    size_t operator()(const pair<size_t, size_t> &x) const
    {
        return hash<long long>()(((long long)x.first) ^ (((long long)x.second) << 32));
    }
};

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class SMatrix
{
private:
    size_t mRows;
    size_t mCols;
    unordered_map<pair<size_t, size_t>, T, HASH> mData;
    float threshold = 1e-5;
    T _zero = 0;

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

    SMatrix()
    {
        mRows = mCols = 0;
    };

    SMatrix(size_t r, size_t c) : mRows(r), mCols(c){};

    SMatrix(std::vector<std::vector<T>> v)
    {
        mRows = v.size();
        mCols = v[0].size();
        for (size_t i = 0; i < mRows; i++)
        {
            for (size_t j = 0; j < mCols; j++)
            {
                if (abs(v[i][j]) > threshold)
                    mData[{i, j}] = v[i][j];
            }
        }
    }

    size_t size() const
    {
        return mRows * mCols;
    }

    typename unordered_map<pair<size_t, size_t>, T>::iterator begin()
    {
        return mData.begin();
    }

    typename unordered_map<pair<size_t, size_t>, T>::iterator end()
    {
        return mData.end();
    }

    // operators
    friend SMatrix operator*(double x, const SMatrix &B)
    {
        SMatrix<double> A(B.nRows, B.nCols);

        for (auto it = B.begin(); it != B.end(); it++)
        {
            A(it->first.first, it->first.second) = it->second * x;
        }

        return A;
    };

    friend SMatrix operator*(const SMatrix &B, double x)
    {
        return x * B;
    };

    SMatrix operator*=(double x)
    {
        for (auto it = mData.begin(); it != mData.end(); it++)
        {
            it->second = it->second * x;
        }
        return *this;
    };

    friend SMatrix operator/(double x, const SMatrix &B)
    {
        // x / B
        SMatrix<double> A(B.nRows, B.nCols);

        for (auto it = B.begin(); it != B.end(); it++)
        {
            A(it->first.first, it->first.second) = x / it->second;
        }

        return A;
    };

    friend SMatrix operator/(const SMatrix &B, double x)
    {
        SMatrix<double> A(B.nRows, B.nCols);

        for (auto it = B.begin(); it != B.end(); it++)
        {
            A(it->first.first, it->first.second) = it->second / x;
        }

        return A;
    };

    SMatrix operator/=(double x)
    {
        for (auto it = mData.begin(); it != mData.end(); it++)
        {
            it->second = it->second / x;
        }
        return *this;
    };

    // get index
    T operator()(size_t i, size_t j) const
    {
        validateIndexes(i, j);
        if (mData.find({i, j}) != mData.end())
            return mData.at({i, j});
        else
            return 0;
    };

    // set index
    void operator[](pair<pair<size_t, size_t>, double> index)
    {
        if (abs(index.second) > threshold)
        {
            mData[{index.first.first, index.first.second}] = index.second;
        }
    }

    static SMatrix<T> identity(size_t n)
    {
        SMatrix<T> A(n, n);
        for (size_t i = 0; i < n; i++)
        {
            A[{{i, i}, 1}];
        }

        return A;
    };

    SMatrix<T> transpose()
    {
        SMatrix<T> A(mCols, mRows);
        for (auto it = begin(); it != end(); it++)
        {
            A[{{it->first.second, it->first.first}, it->second}];
        }
        return A;
    };

    friend ostream &operator<<(ostream &os, const SMatrix &matrix)
    {
        const int numWidth = 13;
        char fill = ' ';

        for (size_t i = 0; i < matrix.mRows; i++)
        {
            for (size_t j = 0; j < matrix.mCols; j++)
            {
                // the trick to print a table-like structure was stolen from here
                // https://stackoverflow.com/a/14796892
                os << left << setw(numWidth) << setfill(fill) << to_string(matrix(i, j));
            }
            os << endl;
        }

        return os;
    };

    T max()
    {
        T max = operator()(0, 0);
        for (auto it = begin(); it != end(); it++)
        {
            if (max < it->second)
            {
                max = it->second;
            }
        }

        return max;
    }

    T min()
    {
        T min = operator()(0, 0);
        for (auto it = begin(); it != end(); it++)
        {
            if (min > it->second)
            {
                min = it->second;
            }
        }

        return min;
    }

    vector<T> to_vector()
    {
        vector<T> v;
        for (size_t i = 0; i < nRows(); i++)
        {
            for (size_t j = 0; j < nCols(); j++)
            {
                v.push_back(operator()(i, j));
            }
        }
        return v;
    }
};

typedef SMatrix<float> SMatrixF;

#endif