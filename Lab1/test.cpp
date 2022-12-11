#include "Gaussian.hpp"
#include <iostream>
using namespace std;

int main()
{
    GaussianElimination G;
    G.readFile("A.txt", "B.txt");
    G.solve();

    cout << G.n << "\n";

    cout << "A\n";
    G.printA();

    cout << "B\n";
    G.printArr(G.B);

    cout << "X\n";
    G.printArr(G.X);
}
