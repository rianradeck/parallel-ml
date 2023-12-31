#include <thread>
#include <utility>
#include <omp.h>

struct Matrix;
std::ostream& operator<<(std::ostream& os, Matrix& o);

struct Matrix{

    int row, col, size;
    double *matrix;

    Matrix(int row_, int col_, double init = 0) : row(row_), col(col_) {
        matrix = new double[row * col];
        size = row * col;
        for(int i = 0;i < row;i++)
            for(int j = 0;j < col;j++)
                matrix[getIndex(i, j)] = init;
    }

    Matrix(const Matrix &o) {
        row = o.row;
        col = o.col;
        size = o.col * o.row;
        matrix = new double[row * col];
        for(int i = 0;i < row;i++)
            for(int j = 0;j < col;j++)
                matrix[getIndex(i, j)] = o.matrix[getIndex(i, j)];
    }

    Matrix(){}

    /*~Matrix(){
        delete[] matrix;
    }*/

    std::pair<int, int> getIndex(int idx){
        return {idx / col, idx % col};
    }

    int getIndex(int a, int b) const {
		//Row major
        return a * col + b;
    }

    double getElement(int i, int j) const {
        return matrix[getIndex(i, j)];
    }

	void setElement(int i, int j, double val) {
		matrix[getIndex(i, j)] = val;
	}

    Matrix operator*(const Matrix& o) const {
        Matrix ret(row, o.col);
        if(col != o.row){
            std::cerr << "YOU CANNOT MULTIPLY THESE MATRICES!\n";
            return ret;
        }

        for(int i = 0;i < row;i++){
            for(int j = 0;j < o.col;j++){
                double sum = 0;
                for(int k = 0;k < col;k++)
                    sum += getElement(i, k) * o.getElement(k, j);
                ret.matrix[ret.getIndex(i, j)] = sum;
            }
        }
        return ret;
    }

    Matrix operator%(const Matrix& o) const {
        Matrix ret(row, o.col);
        if(col != o.row){
            std::cerr << "YOU CANNOT MULTIPLY THESE MATRICES!\n";
            return ret;
        }

        omp_set_dynamic(0);
        omp_set_num_threads(12);
        #pragma omp parallel for
        for(int idx = 0;idx < ret.size;idx++){
            int i, j;
            i = idx / o.col, j = idx % o.col;

            double sum = 0;
            for(int k = 0;k < col;k++)
                sum += getElement(i, k) * o.getElement(k, j);

            ret.matrix[i * o.col + j] = sum;
        }

        return ret;
    }
};

std::ostream& operator<<(std::ostream& os, Matrix& o){
    for(int i = 0;i < o.row;i++){
        for(int j = 0;j < o.col;j++)
            os << o.getElement(i, j) << " ";
        os << "\n";
    }
    return os;
}

std::istream& operator>>(std::istream& is, Matrix& o)
{
    for(int i = 0;i < o.row;i++){
        for(int j = 0;j < o.col;j++){
            double x;
            is >> x;
            o.matrix[o.getIndex(i, j)] = x;
        }
    }
    return is;
}
