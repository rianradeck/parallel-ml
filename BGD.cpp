#include <bits/stdc++.h>

class gd{
public:
    int n_dim;
    double *w0, *w, lr, *gradient;
    void (*dfx)(double [], double []);

    gd(void (*dfx_)(double [], double []), double w0_[], int n_, double lr_=1e-4) : n_dim(n_), w0(w0_), w(new double[n_dim]), lr(lr_), gradient(new double[n_dim]), dfx(dfx_){
        for(int i = 0;i < n_dim;i++){
            w[i] = w0[i];
        }
    } 

    ~gd(){
        delete[] w;
        delete[] gradient;
    }

    void operator()(int iterations = 1){
        for(int itr = 0;itr < iterations;itr++){
            dfx(w, gradient);
            for(int i = 0;i < n_dim;i++){
                w[i] = w[i] - lr*gradient[i];
            }
        }
    }
};

double fx(double x[]){
    double y = 2 * x[0] * x[0] + x[0] + 2 +
               3 * x[1] * x[1] + 2 * x[1] + 1;
    return y;
}

double predict(double w[], double X[], int n_dim){
    double ret = 0;
    for(int i = 0;i < n_dim;i++)
        ret += w[i] * X[i];
    return ret;
}


int main(){
    ios_base::sync_with_stdio(false), cin.tie(0);
    // const int n_dim = 8; // dimension of data + 1 (bias)
    // const int n_data = 208;

    // double X[208][8];
    // double y[208];
    // for(int i = 0;i < n_data;i++){
    //     X[i][0] = 1;
    //     for(int j = 1;j < n_dim;j++)
    //         cin >> X[i][j];
    // }
    // for(int i = 0;i < n_data;i++)
    //     cin >> y[i];

    // double w_init[8] = {0., 0., 0., 0., 0, 0, 0, 0};
    const int n_dim = 2;
    double w_init[2] = {0, 0};
    auto dfx = [](double w[], double gradient[]){
        // for(int i = 0;i < n_dim;i++)
        //     gradient[i] = 0;

        // for(int i = 0;i < n_data;i++){
        //     double prd = predict(w, X[i], n_dim);
        //     gradient[0] += 2 / n_data * (prd - y[i]);
        // }

        // for(int i = 1;i < n_dim;i++)
        //     for(int j = 0;j < n_data;j++){
        //         double prd = predict(w, X[J], n_dim);
        //         gradient[i] += 2 / n_data * X[j][i - 1] * (prd - y[i])
        //     }
        gradient[0] = 4 * w[0] + 1;
        gradient[1] = 6 * w[1] + 2;
    };

    gd BGD(dfx, w_init, n_dim, 1e-1);
    BGD(1e7);
    for(int i = 0;i < n_dim;i++)
        std::cout << BGD.w[i] << "\n";
    return 0;
}