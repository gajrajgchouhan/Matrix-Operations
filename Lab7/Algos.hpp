#ifndef ALGOS_HPP
#define ALGOS_HPP

#include "Matrix.hpp"

class Algos
{
private:
    void backSubs(size_t n, MatrixF &X, MatrixF &B, MatrixF &A)
    {
        for (int i = n - 1; i > -1; i--)
        {
            if (i == n - 1)
            {
                X(i, 1) = B(i, 1) / A(i, i);
            }
            else
            {
                float s = 0;

                for (int j = i + 1; j < n; j++)
                {
                    s += A(i, j) * X(j, 1);
                }
                X(i, 1) = (B(i, 1) - s) / A(i, i);
            }
        }
    }

    void swapRows(MatrixF &A, size_t n, int col, int mx_row)
    {
        // swaprow
        for (int col2 = 0; col2 < n; col2++)
        {

            float tmp1 = A(col, col2);
            float tmp2 = A(mx_row, col2);
            A(col, col2) = tmp2;
            A(mx_row, col2) = tmp1;
        }
    }

public:
    Algos(){};
    ~Algos(){};

    tuple<MatrixF, float> PowerMethod(MatrixF A)
    {
        float threshold = 1e-3;
        size_t r = A.nRows();
        MatrixF guess = MatrixF::ones(r, 1);
        MatrixF old_guess = guess.copy();
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
        }
        MatrixF eigval = (A * guess) / guess;
        MatrixF eigvec = guess;
        float eigvalf = eigval.max();
        eigvec /= eigvec.norm();
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

    MatrixF GaussianElimination(size_t n, MatrixF A, MatrixF X)
    {

        MatrixF B(n, 1);

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

            swapRows(A, n, col, mx_row);

            float tmp1 = B(col, 1);
            float tmp2 = B(mx_row, 1);

            B(mx_row, 1) = tmp1;
            B(col, 1) = tmp2;

            for (int row = col + 1; row < n; row++)
            {
                // make starting to row zero
                float m = A(row, col) / A(col, col);
                for (int i = col; i < n; i++)
                {
                    A(row, i) = A(row, i) - (A(col, i) * m);
                }
                B(row, 1) = B(row, 1) - B(col, 1) * m;
            }
        }
        backSubs(n, X, B, A);
        return B;
    };
};

#endif //ALGOS_HPP