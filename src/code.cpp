#include <iostream>
//#include "Matrix.h"
#include "testSuite.h"
#include <omp.h>

using namespace std;

int main(){
	//testSuite::testLinearRegression();
    Matrix A(10000, 10000, 1);
    Matrix B(10000, 1, 2);
    //cin >> A >> B;
    //cout << A << "\n" << B << "\n";
    Matrix C = A % B;
    cout << C.getElement(0, 0);
}
