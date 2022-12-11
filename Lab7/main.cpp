#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include "Matrix.hpp"
#include "Algos.hpp"
using namespace std;

Matrix<float> generateSymmetric2DMat(uniform_real_distribution<> dist, mt19937 e2, size_t n)
{
    Matrix<float> arr(n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i >= j)
            {
                arr(j, i) = arr(i, j) = (float)dist(e2);
            }
        }
    }
    return arr;
}

template <typename T>
void save2DMat(Matrix<T> &arr, int n, string filename)
{
    ofstream myfile(filename);
    myfile.setf(ios::fixed);
    myfile.precision(10);

    if (myfile.is_open())
    {

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                myfile << i + 1 << " ";
            }
        }
        myfile << "\n";

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                myfile << j + 1 << " ";
            }
        }
        myfile << "\n";

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                myfile << arr(i, j) << " ";
            }
        }
        myfile << "\n";
        myfile.close();
    }

    else
    {
        cout << "Unable to open file";
    }
}

Matrix<float> load2DMat(int n, string filename)
{

    ifstream myfile(filename);
    string str;
    Matrix<float> arr(n);
    int line = 0,
        i = 0,
        row[n * n],
        col[n * n];

    while (myfile >> str)
    {
        switch (line)
        {
        case 0:
            row[i] = stoi(str) - 1;
            break;
        case 1:
            col[i] = stoi(str) - 1;
            break;
        case 2:
            arr(row[i], col[i]) = stof(str);
            break;
        }

        i++;
        if (i == n * n)
        {
            i = 0;
            line++;
        }
    }

    return arr;
}

int main()
{
    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(0, 1);

    cout << "Question 1" << endl;
    cout << "Enter n: ";
    int n;
    cin >> n;

    cout << "Generating symmetric 2D Matrix of size " << n << endl;
    MatrixF arr = generateSymmetric2DMat(dist, e2, n);
    cout << "Generated....printing..." << endl;
    cout << endl;

    cout << arr << endl;
    cout << "Saving to file b.txt" << endl;
    save2DMat(arr, n, "b.txt");

    cout << "Loading from file b.txt" << endl;
    arr = load2DMat(n, "b.txt");
    cout << "Loaded...printing" << endl
         << arr << endl;

    Algos algos;

    cout << "Power Method" << endl;
    MatrixF eigvec, eigval;
    float eigvalf;
    tie(eigvec, eigvalf) = algos.PowerMethod(arr);

    cout << "Dominant eigval" << endl
         << eigvalf << endl;
    cout << "Dominant eigvec" << endl
         << eigvec << endl;

    cout << "Eigenvalue from QR method" << endl;
    tie(eigvec, eigval) = algos.EigfromQR(arr);

    cout << "eigval" << endl
         << eigval << endl
         << "eigvec" << endl
         << eigvec << endl;

    cout << "SVD" << endl;
    MatrixF U, S, V;
    tie(U, S, V) = algos.SVD(arr);

    cout << "U" << endl
         << U << endl
         << "V" << endl
         << V << endl
         << "S" << endl
         << S << endl;

    save2DMat(U, U.nRows(), "U.txt");
    save2DMat(V, V.nRows(), "V.txt");
    save2DMat(S, S.nRows(), "S.txt");
}
