struct Matrix{

    int row, col;
    double *matrix;

    Matrix(int row_, int col_, double init = 0) : row(row_), col(col_) {
        matrix = new double[row * col];
        for(int i = 0;i < row;i++)
            for(int j = 0;j < col;j++)
                matrix[get_index(i, j)] = init;
    }

    ~Matrix(){
        delete[] matrix;
    }

    pair<int, int> get_index(int idx){
        return {idx / col, idx % col};
    }

    int get_index(int a, int b) const {
        return a * col + b;
    }

    double get_element(int i, int j) const {
        return matrix[get_index(i, j)];
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
                    sum += get_element(i, k) * o.get_element(k, j);
                ret.matrix[ret.get_index(i, j)] = sum;
            }
        }
        return ret;
    }
};

std::ostream& operator<<(std::ostream& os, Matrix& o){
    for(int i = 0;i < o.row;i++){
        for(int j = 0;j < o.col;j++)
            os << o.get_element(i, j) << " ";
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
            o.matrix[o.get_index(i, j)] = x;
        }
    }
    return is;
}