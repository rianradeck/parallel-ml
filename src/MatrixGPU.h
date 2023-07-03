#include <thread>
#include <utility>


struct MatrixGPU;
__global__ void multBlock(double *C, double *A, double *B, int Arow, int Acol, int Brow, int Bcol, int size);
std::ostream& operator<<(std::ostream& os, MatrixGPU& o);

struct MatrixGPU{

    int row, col, size;
    double *matrix;

    MatrixGPU(int row_, int col_, double init = 0) : row(row_), col(col_) {
        matrix = new double[row * col];
        size = row * col;
        for(int i = 0;i < row;i++)
            for(int j = 0;j < col;j++)
                matrix[getIndex(i, j)] = init;
    }

    MatrixGPU(const MatrixGPU &o) {
        row = o.row;
        col = o.col;
        size = o.col * o.row;
        matrix = new double[row * col];
        for(int i = 0;i < row;i++)
            for(int j = 0;j < col;j++)
                matrix[getIndex(i, j)] = o.matrix[getIndex(i, j)];
    }

    MatrixGPU(){}

    /*~MatrixGPU(){
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

    MatrixGPU operator*(const MatrixGPU& o) const {
        MatrixGPU ret(row, o.col);
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

    MatrixGPU operator%(const MatrixGPU& o) const {
        MatrixGPU ret(row, o.col);
        if(col != o.row){
            std::cerr << "YOU CANNOT MULTIPLY THESE MATRICES!\n";
            return ret;
        }


        double *d_A, *d_B, *d_C;
        
        auto t = std::chrono::high_resolution_clock::now();
        cudaMalloc((void **) &d_A, sizeof(double) * size);
        cudaMalloc((void **) &d_B, sizeof(double) * o.size);
        cudaMalloc((void **) &d_C, sizeof(double) * ret.size);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto d1 = t1 - t;
        // std::cerr << "Time to allocate memory on device: " << d1.count() / 1e9 << std::endl;

        cudaMemcpy(d_A, matrix, sizeof(double) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, o.matrix, sizeof(double) * o.size, cudaMemcpyHostToDevice);
        
        int threads_per_block = 128;
        int blocks = (ret.size + threads_per_block - 1) / threads_per_block;
        multBlock<<<blocks, threads_per_block>>>(d_C, d_A, d_B, row, col, o.row, o.col, ret.size);
        cudaDeviceSynchronize();

        cudaMemcpy(ret.matrix, d_C, sizeof(double) * ret.size, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // for(int i = 0;i < ret.size();i++)
        //     ret.matrix[i] = C[i];

        return ret;
    }
};

std::ostream& operator<<(std::ostream& os, MatrixGPU& o){
    for(int i = 0;i < o.row;i++){
        for(int j = 0;j < o.col;j++)
            os << o.getElement(i, j) << " ";
        os << "\n";
    }
    return os;
}

std::istream& operator>>(std::istream& is, MatrixGPU& o)
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


__global__ void multBlock(double *C, double *A, double *B, int Arow, int Acol, int Brow, int Bcol, int size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < size){
        int i, j;
        i = idx / Bcol, j = idx % Bcol;

        double sum = 0;
        for(int k = 0;k < Acol;k++)
            sum += A[i * Acol + k] * B[k * Bcol + j];

        C[i * Bcol + j] = sum;
    }
}