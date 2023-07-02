#include <bits/stdc++.h>
#include "Matrix.h"

using namespace std;

int main(){
    Matrix A(2, 3);
    Matrix B(3, 2);
    cin >> A >> B;
    cout << A << "\n" << B << "\n";
    Matrix C = A * B;
    cout << C;
}