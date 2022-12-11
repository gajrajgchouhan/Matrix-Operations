#ifndef GAUSSIAN_ELI
#define GAUSSIAN_ELI

#include <string>
#include <fstream>
#include <iostream>
using namespace std;

// for (int i = 0; i < n; i++)
// {
//     for (int j = 0; j < n; j++)
//     {
//         setRC(this->A, i, j, A[i][j]);
//     }
// }

// for (int i = 0; i < n; i++)
// {
//     *(this->B + i) = B[i];
// }

class GaussianElimination
{
private:
    float getRC(float *XX, int r, int c)
    {
        return *(XX + (r * n) + c);
    }

    void setRC(float *XX, int r, int c, float val)
    {
        *((XX + (r * n)) + c) = val;
    }

    void backSubs()
    {
        for (int i = n - 1; i > -1; i--)
        {
            if (i == n - 1)
            {
                *(X + i) = *(B + i) / getRC(A, i, i);
            }
            else
            {
                float s = 0;

                for (int j = i + 1; j < n; j++)
                {
                    s += getRC(A, i, j) * X[j];
                }
                *(X + i) = (B[i] - s) / getRC(A, i, i);
            }
        }
    }

    void swapRows(int col, int mx_row)
    {
        // swaprow
        for (int col2 = 0; col2 < n; col2++)
        {

            float tmp1 = getRC(A, col, col2);
            float tmp2 = getRC(A, mx_row, col2);
            setRC(A, col, col2, tmp2);
            setRC(A, mx_row, col2, tmp1);
        }
    }

public:
    int n;
    float *A;
    float *B;
    float *X;
    void readFile(string Afile, string Bfile)
    {
        ifstream input_A(Afile);
        ifstream input_B(Bfile);

        int n;
        input_A >> n;
        this->n = n;

        this->A = (float *)malloc(sizeof(float) * n * n);
        this->B = (float *)malloc(sizeof(float) * n);
        this->X = (float *)malloc(sizeof(float) * n);

        float tmp;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                input_A >> tmp;
                setRC(A, i, j, tmp);
            }
        }

        for (int i = 0; i < n; i++)
        {
            input_B >> tmp;
            *(B + i) = tmp;
        }
    };

    void solve()
    {
        // Gaussian

        for (int col = 0; col < n; col++)
        {
            // PIVOT

            float mx = getRC(A, col, col);
            int mx_row = col;
            for (int row = col + 1; row < n; row++)
            {
                if (getRC(A, row, col) > mx)
                {
                    mx = getRC(A, row, col);
                    mx_row = row;
                }
            }

            swapRows(col, mx_row);

            float tmp1 = *(B + col);
            float tmp2 = *(B + mx_row);

            *(B + mx_row) = tmp1;
            *(B + col) = tmp2;

            for (int row = col + 1; row < n; row++)
            {
                // make starting to row zero
                float m = getRC(A, row, col) / getRC(A, col, col);
                for (int i = col; i < n; i++)
                {
                    setRC(A, row, i, getRC(A, row, i) - (getRC(A, col, i) * m));
                }
                *(B + row) = *(B + row) - *(B + col) * m;
            }
        }
        backSubs();
    };

    void printA()
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                cout << getRC(A, i, j) << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    };

    void printArr(float *X)
    {
        for (int i = 0; i < n; i++)
        {
            cout << *(X + i) << " ";
        }
        cout << "\n";
    };
};

#endif