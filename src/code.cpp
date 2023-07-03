#include <iostream>
//#include "Matrix.h"
#include "testSuite.h"
#include <chrono>

using namespace std;

int main(){
	testSuite::testLinearRegression();
    // Matrix A(10000, 10000, 1);
    // Matrix B(10000, 1, 2);
    // //cin >> A >> B;
    // //cout << A << "\n" << B << "\n";
    // auto t1 = chrono::high_resolution_clock::now();
    // Matrix C = A % B;
    // // cout << C.getElement(0, 0);
    // auto t2 = chrono::high_resolution_clock::now();
    // C = A * B;
    // // cout << C.getElement(0, 0);
    // auto t3 = chrono::high_resolution_clock::now();

    // auto s1 = t2 - t1, s2 = t3 - t2;

    // cout << "Parallel: " << s1.count() / 1e9 << " s\nSerial: " << s2.count() / 1e9 << " s\n";
}
