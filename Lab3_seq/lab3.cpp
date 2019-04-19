//
// Created by rahul on 5/4/19.
//
#include "lab3_cuda.h"
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <fstream>


using namespace std;

#define TOLERANCE 0.001
#define JACOBI_UPDATE_TOLERANCE 0.001

double **S; //Symmetric matrix (input)
double  *e; //eigenvalues
double **E; //eigenvectors
int  *ind;
bool *changed;
int  state;
int  N;


double** mat_transpose(double** A, int Am, int An) {
    double **B;
    B = (double**)malloc(__SIZEOF_POINTER__*An);
    for (int i=0; i<An; i++)
        B[i] = (double*)malloc(__SIZEOF_DOUBLE__*Am);

    for (int i=0; i<Am; i++){
        for (int j=0; j<An; j++){
            B[j][i] = A[i][j];
        }
    }

    return B;
}

double** mat_mul(double** A, int Am, int An,
                 double** B, int Bm, int Bn){
    double **C;
    C = (double**)malloc(__SIZEOF_POINTER__*Am);
    for (int i=0; i<Am; i++)
        C[i] = (double*)malloc(__SIZEOF_DOUBLE__*Bn);

    for (int i=0; i<Am; i++){
        for (int j=0; j<Bn; j++){
            C[i][j] = 0;
            for (int k=0; k<An; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

int maxind(int k) {
    int m = k+1;

    for (int i = k+2; i < N; i++){
        if (fabs(S[k][i]) > fabs(S[k][m])){
            m = i;
        }
    }

    return m;
}

void update(int k, double t) {
    double ek_prev = e[k];
    e[k] = ek_prev + t;

    if (e[k] < 0) e[k] = 0;

    if (changed[k] && (ek_prev - e[k]) < JACOBI_UPDATE_TOLERANCE) {
        changed[k] = false;
        state = state - 1;
    }
    else if ((! changed[k]) && (ek_prev - e[k]) > JACOBI_UPDATE_TOLERANCE) {
        changed[k] = true;
        state = state + 1;
    }
}

void rotate(int k, int l, int i, int j, double c, double s,
            bool eigenvectors){
    double** mat1;
    double** mat2;
    double** mat3;

    mat1 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat1[0] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[1] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[0][0] = c; mat1[0][1] = -s;
    mat1[1][0] = s; mat1[1][1] = c;

    mat2 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat2[0] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    mat2[1] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    if (eigenvectors){
        mat2[0][0] = E[i][k];
        mat2[1][0] = E[i][l];
    }
    else {
        mat2[0][0] = S[k][l];
        mat2[1][0] = S[i][j];
    }

    mat3 = mat_mul(mat1, 2, 2, mat2, 2, 1);

    if (eigenvectors){
        E[i][k] = mat3[0][0];
        E[i][l] = mat3[1][0];
    }
    else{
        S[k][l] = mat3[0][0];
        S[i][j] = mat3[1][0];
    }

    free(mat1[0]);
    free(mat1[1]);
    free(mat1);
    free(mat2[0]);
    free(mat2[1]);
    free(mat2);
    free(mat3[0]);
    free(mat3[1]);
    free(mat3);
}

void print_matrix(double** A, int Am, int An) {
    cout << "[";
    for (int i=0; i<Am; i++){
        if (i>0)
            cout<<" ";
        cout<<"[";
        for (int j=0; j<An-1; j++){
            cout << A[i][j] << ", ";
        }
        if (i < Am-1)
            cout << A[i][An-1] << "]" << endl;
    }
    cout << A[Am-1][An-1] << "]]" << endl;
}

void print_vector(double* A, int An) {
    cout << "[";
    for(int i=0; i<An-1; i++)
        cout << A[i] << ",";
    cout << A[An-1] << "]" << endl;
}

void init_jacobi() {
    E = (double**)malloc(__SIZEOF_POINTER__*N);
    for (int i=0; i<N; i++){
        E[i] = (double*)malloc(__SIZEOF_DOUBLE__*N);
        for (int j=0; j<N; j++){
            E[i][j] = 0;
        }
        E[i][i] = 1;
    }

    state = N;

    e = (double*)malloc(__SIZEOF_DOUBLE__*N);
    ind = (int*)malloc(__SIZEOF_INT__*N);
    changed = (bool*)malloc(sizeof(bool)*N);

    for (int k=0; k<N; k++){
        ind[k]     = maxind(k);
        e[k]       = S[k][k];
        changed[k] = true;
    }
}

void Jacobi(double **input_matrix, int n,
            double **eigenvalues, double ***eigenvectors) {
    N = n;
    S = input_matrix;

    init_jacobi();

    while(state != 0){
        int m = 0;

        for (int k=1; k<N-1; k++){
            if (fabs(S[k][ind[k]]) > fabs(S[m][ind[m]])){
                m = k;
            }
        }

        int k = m;
        int l = ind[m];
        double p = S[k][l];
        double y = (e[l] - e[k]) / 2.0;
        double d = fabs(y) + sqrt(p*p + y*y);
        double r = sqrt(p*p + d*d);
        double c = d / r;
        double s = p / r;
        double t = (p*p) / d;

        if (y < 0.0) { s = -s; t = -t; }

        S[k][l] = 0.0;
        update(k, -t);
        update(l, t);

        for (int i=0; i<k; i++)  { rotate(i, k, i, l, c, s, false); }
        for (int i=k+1; i<l; i++){ rotate(k, i, i, l, c, s, false); }
        for (int i=l+1; i<N; i++)  { rotate(k, i, l, i, c, s, false); }

        for (int i=0; i<N; i++){
            rotate(k, l, i, i, c, s, true);
        }

        ind[k] = maxind(k);
        ind[l] = maxind(l);
    }

    *eigenvalues = e;
    *eigenvectors = E;
}


void SVD_and_PCA (int M,
                  int N,
                  double* D,
                  double** U,
                  double** SIGMA,
                  double** V_T,
                  double** D_HAT,
                  int *K,
                  int retention) {

    double **D_new, **D_T, **U_new, **sigma, **V_new, **sigma_inv, **intermediate, **D_hat;
    double **prod, *eigenvalues, **eigenvectors;

    D_new = (double**)malloc(sizeof(double*)*M);
    for (int i=0; i<M; i++)
        D_new[i] = (double*)malloc(sizeof(double)*N);

    sigma = (double**)malloc(sizeof(double*)*M);
    for (int i=0; i<M; i++)
        sigma[i] = (double*)malloc(sizeof(double)*N);

    V_new = (double**)malloc(sizeof(double*)*N);
    for (int i=0; i<N; i++)
        V_new[i] = (double*)malloc(sizeof(double)*N);

    sigma_inv = (double**)malloc(sizeof(double*)*N);
    for (int i=0; i<N; i++)
        sigma_inv[i] = (double*)malloc(sizeof(double)*M);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            D_new[i][j]=D[N*i+j];
        }
    }

    D_T = mat_transpose(D_new, M, N);
    prod = mat_mul(D_T, N, M, D_new, M, N);


    Jacobi(prod, N, &eigenvalues, &eigenvectors);

    double sqrt_eigen_values[N];

    for (int i = 0; i <N ; ++i) {
        sqrt_eigen_values[i]=sqrt(eigenvalues[i]);
    }

    double sorted_singular_values[N];    // taking only first N eigen values in decending order.

    // Sorting the eigen values and computing the matrix V

    for (int i = 0; i <N ; ++i) {
        double max=0;
        int index=0;
        for (int j = 0; j <N ; ++j) {
            if (sqrt_eigen_values[j]>max){
                max=sqrt_eigen_values[j];
                index=j;
            }
        }
        sorted_singular_values[i]=max;
        for (int j = 0; j <N ; ++j) {
            V_new[j][i]=eigenvectors[j][index];
        }
        sqrt_eigen_values[index]=0;
    }

    for (int i = 0; i <M ; ++i) {
        for (int j = 0; j <N ; ++j) {
            sigma[i][j]=0;
        }
    }



    // Computing matrix sigma
    for (int i = 0; i <N ; ++i) {
        sigma[i][i]=sorted_singular_values[i];

    }
// Computing matrix sigma_inv
    for (int i = 0; i <N ; ++i) {
        for (int j = 0; j <M ; ++j) {
            if (i!=j) sigma_inv[i][j]=0;
            else sigma_inv[i][j]=(1/sorted_singular_values[j]);
        }
    }

    // Computing U = D_new_T V Sigma_inv
    intermediate=mat_mul(D_new,M,N,V_new,N,N);
    U_new = mat_mul(intermediate,M,N,sigma_inv,N,M);

    for (int i = 0; i <N ; ++i) {
        SIGMA[0][i]=sigma[i][i];
    }

    for (int i = 0; i <N ; ++i) {
        for (int j = 0; j <N ; ++j) {
            U[0][N*i+j]=V_new[i][j];
        }
    }

    for (int i = 0; i <M ; ++i) {
        for (int j = 0; j <M ; ++j) {
            V_T[0][M*i+j]=U_new[j][i];
        }
    }

    // PCA Part -

    double sum_eigen_values = 0;

    for (int i = 0; i <N ; ++i) {
        sum_eigen_values+=sigma[i][i]*sigma[i][i];
    }

    int count=0;
    double sum=0;
    for (int i = 0; i <N ; ++i) {
        count+=1;
        sum+=sigma[i][i]*sigma[i][i];
        if ((sum/sum_eigen_values)*100>=retention) break;
    }

    *K=count;

    double **U_After_PCA;
    U_After_PCA = (double**)malloc(sizeof(double*)*N);
    for (int i=0; i<N; i++)
        U_After_PCA[i] = (double*)malloc(sizeof(double)*count);

    for (int i = 0; i <N ; ++i) {
        for (int j = 0; j <count ; ++j) {
            U_After_PCA[i][j]=V_new[i][j];
        }
    }

    D_hat=mat_mul(D_new,M,N,U_After_PCA,N,count);

    D_HAT[0] = (double*) malloc(sizeof(double) * M*count);

    for (int i = 0; i <M ; ++i) {
        for (int j = 0; j <count ; ++j) {
            D_HAT[0][count*i+j]=D_hat[i][j];
        }
    }

//    cout << "\nsingularvalues:" << endl;
//    print_vector(sorted_singular_values, N);
//    cout <<count<< "\n" << endl;
    cout << "\nD_Hat:" << endl;
    print_matrix(D_hat, M, count);

}
