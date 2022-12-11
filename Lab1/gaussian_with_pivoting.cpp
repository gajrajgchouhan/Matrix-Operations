#include <bits/stdc++.h>
using namespace std;

int main()
{
    ifstream input_A("A.txt");
    ifstream input_B("B.txt");

    int n;
    input_A >> n;

    float A[n][n];
    float B[n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            input_A >> A[i][j];
        }
    }

    for (int i = 0; i < n; i++)
    {
        input_B >> B[i];
    }

    // Gaussian

    for (int col = 0; col < n; col++)
    {
        // PIVOT

        float mx = A[col][col];
        int mx_row = col;
        for (int row = col + 1; row < n; row++)
        {
            if (A[row][col] > mx)
            {
                mx = A[row][col];
                mx_row = row;
            }
        }

        float tmp1 = B[col];
        float tmp2 = B[mx_row];

        B[mx_row] = tmp1;
        B[col] = tmp2;

        // swaprow
        for (int col2 = 0; col2 < n; col2++)
        {
            float tmp1 = A[col][col2];
            float tmp2 = A[mx_row][col2];
            A[col][col2] = tmp2;
            A[mx_row][col2] = tmp1;
        }

        for (int row = col + 1; row < n; row++)
        {
            // make starting to row zero
            float m = A[row][col] / A[col][col];
            for (int i = col; i < n; i++)
            {
                A[row][i] -= (A[col][i] * m);
            }
            B[row] -= B[col] * m;
        }
    }

    // Back substitute
    float X[n];

    for (int i = n - 1; i > -1; i--)
    {
        if (i == n - 1)
        {
            X[i] = B[i] / A[i][i];
        }
        else
        {
            float s = 0;

            for (int j = i + 1; j < n; j++)
            {
                s += A[i][j] * X[j];
            }
            X[i] = (B[i] - s) / A[i][i];
        }
    }

    cout << "A\n";
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << A[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    cout << "B\n";
    for (int i = 0; i < n; i++)
    {
        cout << B[i] << "\n";
    }
    cout << "\n";

    cout << "X\n";
    for (int i = 0; i < n; i++)
    {
        cout << X[i] << "\n";
    }
    cout << "\n";
}