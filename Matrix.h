#include <thread>
#include <utility>

const int NUM_TREADS = 4;

struct Matrix;
void multBlock(int block, Matrix *ret, const Matrix &A, const Matrix& B);
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

    ~Matrix(){
        delete[] matrix;
    }

    std::pair<int, int> getIndex(int idx){
        return {idx / col, idx % col};
    }

    int getIndex(int a, int b) const {
        return a * col + b;
    }

    double getElement(int i, int j) const {
        return matrix[getIndex(i, j)];
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


        int NUM_TREADS = 4;
        std::thread t[NUM_TREADS];

        for(int i = 0;i < NUM_TREADS;i++){
            t[i] = std::thread(multBlock, i, &ret, *this, o);
            cout << "Spawned thread " << i << "\n";
        }
        for(int i = 0;i < NUM_TREADS;i++){
            dbg(i);
            t[i].join();
            cout << "Joined " << i << "\n";
        }

        cout << ret << "\n";
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

void multBlock(int block, Matrix *ret, const Matrix &A, const Matrix& B){

    int blockSize = (ret->size + NUM_TREADS - 1) / NUM_TREADS;
    int start = block * blockSize;
    int finish = std::min(start + blockSize, ret->size); 

    for(int idx = start;idx < finish;idx++){
        int i, j;
        std::tie(i, j) = ret->getIndex(idx);

        double sum = 0;
        for(int k = 0;k < A.col;k++)
            sum += A.getElement(i, k) * B.getElement(k, j);
        ret->matrix[ret->getIndex(i, j)] = sum;
    }
}