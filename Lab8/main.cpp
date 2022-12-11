#include <fstream>
#include <string>
#include <iostream>
#include "Algos.hpp"
#include "SMatrix.hpp"
// from lab7 for power and qr method,
#include "Matrix.hpp"
#include "main.hpp"
//

#include <random>
using namespace std;

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

SMatrixF load2DMat(string filename)
{

    ifstream myfile(filename);
    string str;
    vector<vector<float>> v;
    while (myfile >> str)
    {
        vector<float> v1;
        for (auto cell : split(str, ','))
        {

            if (stof(cell) != 0)
            {
                v1.push_back(stof(cell));
            }
            else
            {
                v1.push_back(0);
            }
        }
        v.push_back(v1);
    }
    SMatrixF M(v);

    return M;
}

SMatrixF TriDiagSymmetrix(size_t n)
{
    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(0, 1);
    SMatrixF M(n, n);
    for (size_t i = 0; i < n; i++)
    {
        M[{{i, i}, (float)dist(e2)}];
        if (i != n - 1)
        {
            M[{{i, i + 1}, (float)dist(e2)}];
            M[{{i + 1, i}, (float)dist(e2)}];
        }
    }
    return M;
}

int main()
{
    Algos algos;
    cout << "If random matrix doesnt converge run again, this is due to different thresholdings.\n";
    cout << "loading\n";
    SMatrixF loaded = load2DMat("matrix.txt");
    cout << loaded << endl;

    size_t n;
    cout << "Enter n : ";
    cin >> n;
    SMatrixF tri = TriDiagSymmetrix(n);
    cout << tri << endl;

    SMatrixF eigvector;
    float eigenval;
    cout << "InversePower" << endl;
    tie(eigvector, eigenval) = algos.InversePower(loaded, loaded.nRows(), 1e-4);
    cout << "eigenvector" << endl
         << eigvector << "eigenval " << eigenval << endl
         << endl;

    cout << "RayleighQ" << endl;
    tie(eigvector, eigenval) = algos.RayleighQ(loaded, 1, 1e-2);
    cout << "eigenvector" << endl
         << eigvector << "eigenval " << eigenval << endl
         << endl;

    cout << "InversePower" << endl;
    tie(eigvector, eigenval) = algos.InversePower(tri, tri.nRows(), 1e-2);
    cout << "eigenvector" << endl
         << eigvector << "eigenval " << eigenval << endl
         << endl;

    cout << "RayleighQ" << endl;
    tie(eigvector, eigenval) = algos.RayleighQ(tri, 1, 1e-2);
    cout << "eigenvector" << endl
         << eigvector << "eigenval " << eigenval << endl
         << endl;

    cout << "Comparing with Power and QR" << endl;

    MatrixF part_e(loaded.nRows(), loaded.nCols(), loaded.to_vector());
    cout << "part e \n";
    cout << "Power and QR were taken from Lab7\n";
    algos.PowerMethod(part_e);
    algos.EigfromQR(part_e);
    algos.RayleighQ(loaded, 100000, 1e-4);

    cout << "\nIn short : these Inverse Power / Rayleigh take less iterations as seen\n As seen above print statements.\n";
}
