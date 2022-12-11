#ifndef ALGOS_HPP
#define ALGOS_HPP

#include <map>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <set>
#include <complex>
#include "SMatrix.hpp"
#include "Matrix.hpp"
using namespace std;

class Algos
{
private:
    SMatrixF backSubs(size_t n, SMatrixF &X, SMatrixF &B, SMatrixF &A)
    {
        for (int i = n - 1; i > -1; i--)
        {
            if (i == n - 1)
            {
                X[{{i, 0}, B(i, 0) / A(i, i)}];
            }
            else
            {
                float s = 0;

                for (int j = i + 1; j < n; j++)
                {
                    s += A(i, j) * X(j, 0);
                }
                X[{{i, 0}, (B(i, 0) - s) / A(i, i)}];
            }
        }
        return X;
    }

    SMatrixF swapRows(SMatrixF &A, size_t n, int col, int mx_row)
    {
        // swaprow
        for (int col2 = 0; col2 < n && col != mx_row; col2++)
        {

            float tmp1 = A(col, col2);
            float tmp2 = A(mx_row, col2);
            A[{{col, col2}, tmp2}];
            A[{{mx_row, col2}, tmp1}];
        }
        return A;
    }

    SMatrixF multiply(SMatrixF A, SMatrixF B)
    {
        if (A.nCols() != B.nRows())
        {
            throw invalid_argument("matrix incompatible");
        }
        SMatrixF C(A.nRows(), B.nCols());
        for (size_t i = 0; i < A.nRows(); i++)
        {
            for (size_t j = 0; j < A.nCols(); j++)
            {
                for (size_t k = 0; k < B.nCols(); k++)
                {
                    float s = 0;
                    for (size_t l = 0; l < B.nRows(); l++)
                    {
                        s += A(i, l) * B(l, k);
                    }
                    C[{{i, k}, s}];
                }
            }
        }
        return C;
    }

public:
    SMatrixF GaussianElimination(size_t n, SMatrixF A, SMatrixF B)
    {

        SMatrixF X(n, 1);
        for (int col = 0; col < n; col++)
        {
            // PIVOT

            float mx = A(col, col);
            int mx_row = col;
            for (int row = col + 1; row < n; row++)
            {
                if (A(row, col) > mx)
                {
                    mx = A(row, col);
                    mx_row = row;
                }
            }

            A = swapRows(A, n, col, mx_row);
            float tmp1 = B(col, 0);
            float tmp2 = B(mx_row, 0);

            B[{{mx_row, 0}, tmp1}];
            B[{{col, 0}, tmp2}];

            for (int row = col + 1; row < n; row++)
            {
                // make starting to row zero
                float m = A(row, col) / A(col, col);
                for (int i = col; i < n; i++)
                {
                    A[{{row, i}, A(row, i) - (A(col, i) * m)}];
                }
                B[{{row, 0}, B(row, 0) - B(col, 0) * m}];
            }
        }

        X = backSubs(n, X, B, A);
        return X;
    };
    tuple<SMatrixF, float> InversePower(SMatrixF M, size_t n, float threshold)
    {
        SMatrixF x(M.nRows(), 1);
        for (size_t i = 0; i < M.nRows(); i++)
        {
            x[{{i, 0}, 1}];
        }

        SMatrixF old_x;

        size_t iterations = 0;
        while (1)
        {
            old_x = x;

            x = GaussianElimination(n, M, x);
            x /= x(0, 0);

            float s = 0;
            for (size_t i = 0; i < x.nRows(); i++)
            {
                float y = x(i, 0) - old_x(i, 0);
                if (abs(y) < threshold)
                {
                    s++;
                }
            }

            if (s == n)
            {
                break;
            }
            iterations++;
        }
        // Ax' = x
        auto mul = multiply(M, x);
        float eigenval = mul(0, 0) / x(0, 0);
        x /= norm(x);
        cout << "iterations done in InversePower " << iterations << endl;
        return {x, eigenval};
    };

    float norm(SMatrixF A)
    {
        float s = 0;
        for (auto i : A)
        {
            s += i.second * i.second;
        }
        return sqrt(s);
    }

    tuple<SMatrixF, float> RayleighQ(SMatrixF M, float mu, float threshold)
    {
        SMatrixF x(M.nRows(), 1);
        SMatrixF M2 = M;
        for (size_t i = 0; i < M.nRows(); i++)
        {
            x[{{i, 0}, 1}];
        }
        float old_mu = mu;
        size_t iterations = 0;
        while (1)
        {
            old_mu = mu;
            M2 = M;
            // M - mu * I
            for (size_t i = 0; i < M.nRows(); i++)
            {
                M2[{{i, i}, M2(i, i) - mu}];
            }
            x = GaussianElimination(M2.nRows(), M2, x);
            x /= norm(x);

            SMatrixF tmp = multiply(M, x);
            tmp = multiply(x.transpose(), tmp);

            if (tmp.nRows() != 1 || tmp.nCols() != 1)
            {
                throw invalid_argument("something went wrong");
            }

            mu = tmp(0, 0);
            // cerr << abs(old_mu - mu) << endl;
            if (abs(old_mu - mu) < threshold)
            {
                break;
            }
            iterations++;
        }
        cout << "iterations done in RayLeigh " << iterations << endl;
        return {x, mu};
    };

    tuple<MatrixF, float> PowerMethod(MatrixF A)
    {
        float threshold = 1e-3;
        size_t r = A.nRows();
        MatrixF guess = MatrixF::ones(r, 1);
        MatrixF old_guess = guess.copy();
        size_t iterations = 0;
        while (1)
        {
            guess = A * guess;
            guess /= guess.max();
            float error = (guess - old_guess).norm();
            if (error < threshold)
            {
                break;
            }
            else
            {
                old_guess = guess.copy();
            }
            iterations++;
        }
        MatrixF eigval = (A * guess) / guess;
        MatrixF eigvec = guess;
        float eigvalf = eigval.max();
        eigvec /= eigvec.norm();
        cout << "iterations done in Power " << iterations << endl;
        return {eigvec, eigvalf};
    }

    tuple<MatrixF, MatrixF> EigfromQR(MatrixF A, size_t max_iter = 100)
    {
        MatrixF eigvec = MatrixF::ones(A.nRows(), 1).asDiagonal();
        for (size_t i = 0; i < max_iter; i++)
        {
            MatrixF q, r;
            tie(q, r) = QR(A);
            A = r * q;
            eigvec *= q;
        }
        return {eigvec, A.diagonal()};
    }

    tuple<MatrixF, MatrixF, MatrixF> SVD(MatrixF A)
    {
        size_t m = A.nRows(), n = A.nCols();
        size_t r = min(m, n);
        MatrixF S = MatrixF::zeros(m, n);
        MatrixF helper = A.transpose() * A;
        MatrixF eigenvalues1, eigenvalues2, eigenvectors;
        tie(eigenvectors, eigenvalues1) = EigfromQR(helper);
        tie(eigenvalues1, eigenvectors) = MatrixF::eigsort(eigenvalues1, eigenvectors);

        auto V = eigenvectors.copy();
        size_t j = 0;
        for (auto i : eigenvalues1)
        {
            if (j == r)
            {
                break;
            }
            else
            {
                S(j, j) = sqrt(i);
                j++;
            }
        }
        helper = A * A.transpose();
        tie(eigenvectors, eigenvalues2) = EigfromQR(helper);
        tie(eigenvalues2, eigenvectors) = MatrixF::eigsort(eigenvalues2, eigenvectors);

        auto U = eigenvectors.copy();

        S = U.transpose() * A * V;
        return {U, S, V};
    }

    tuple<MatrixF, MatrixF> QR(MatrixF A)
    {
        size_t n = A.nRows(), k = A.nCols();
        MatrixF Q = MatrixF::zeros(A.nRows(), A.nCols()),
                R = MatrixF::zeros(A.nCols(), A.nCols());
        for (size_t i = 0; i < k; i++)
        {
            MatrixF s = MatrixF::zeros(A.nRows(), 1);
            for (size_t j = 0; j < i; j++)
            {
                R(j, i) = (float)(Q.getColumn(j).transpose() * A.getColumn(i));
                s += (R(j, i) * Q.getColumn(j));
            }
            float norm = (A.getColumn(i) - s).norm();
            Q.setColumn(i, (A.getColumn(i) - s) / norm);
            R(i, i) = norm;
        }
        return {Q, R};
    }
};

#endif