#ifndef MAIN_HPP
#define MAIN_HPP

#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include "Matrix.hpp"
#include "Algos.hpp"
using namespace std;

class MAIN
{
private:
    vector<string> split(const string &s, char seperator)
    {
        vector<string> output;

        string::size_type prev_pos = 0, pos = 0;

        while ((pos = s.find(seperator, pos)) != string::npos)
        {
            string substring(s.substr(prev_pos, pos - prev_pos));

            output.push_back(substring);

            prev_pos = ++pos;
        }

        output.push_back(s.substr(prev_pos, pos - prev_pos)); // Last word

        return output;
    }

public:
    static Matrix<float> generateSymmetric2DMat(uniform_real_distribution<> dist, mt19937 e2, size_t n)
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
    static void save2DMat(Matrix<T> &arr, int n, string filename)
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

    static Matrix<float> load2DMat(int n, string filename)
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
};

// int main()
// {

//     Algos algos;

//     cout << "Power Method" << endl;
//     MatrixF eigvec, eigval;
//     float eigvalf;
//     tie(eigvec, eigvalf) = algos.PowerMethod(arr);

//     cout << "Dominant eigval" << endl
//          << eigvalf << endl;
//     cout << "Dominant eigvec" << endl
//          << eigvec << endl;

//     cout << "Eigenvalue from QR method" << endl;
//     tie(eigvec, eigval) = algos.EigfromQR(arr);

//     cout << "eigval" << endl
//          << eigval << endl
//          << "eigvec" << endl
//          << eigvec << endl;

//     cout << "SVD" << endl;
//     MatrixF U, S, V;
//     tie(U, S, V) = algos.SVD(arr);

//     cout << "U" << endl
//          << U << endl
//          << "V" << endl
//          << V << endl
//          << "S" << endl
//          << S << endl;

//     save2DMat(U, U.nRows(), "U.txt");
//     save2DMat(V, V.nRows(), "V.txt");
//     save2DMat(S, S.nRows(), "S.txt");
// }
#endif