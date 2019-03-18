#include <malloc.h>
#include <omp.h>
#include <math.h>
#include <algorithm>

void reverse_array(double* a, int N){
    double* temp = new double[N];
    for(int i = 0; i < N; i++)
        temp[i] = a[i];
    for(int i = 0; i < N; i++)
        a[i] = temp[N-i-1];
}

double** empty_matrix(int m, int n){
    double** A = new double*[m];
    for(int i = 0; i < m; i++){
        A[i] = new double[n];
        for(int j = 0; j < n; j++)  
            A[i][j] = 0;
    }
    return A;
}

void copy_matrix(double** to, double** from, int n, int m){
    #pragma omp parallel for
    for(int i = 0; i < m; i++){
        #pragma omp parallel for
        for(int j = 0; j < n; j++)  
            to[i][j] = from[i][j];
    }
}

double** diagonal_matrix(int n){
    double** A = new double*[n];
    for(int i = 0; i < n; i++){
        A[i] = new double[n];
        for(int j = 0; j < n; j++)  
            A[i][j] = (i == j) ? 1 : 0;
    }
    return A;
}

double dot(double** A, double** B, int cola, int colb, int N){
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < N; i++){
        sum += A[i][cola] * B[i][colb];
    }
    return sum;
}

double norm(double** A, int cola, int N){
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < N; i++){
        sum += A[i][cola] * A[i][cola];
    }
    return sqrt(sum);
}

double norm_vector(double* A, double N){
    double sum = 0;
    // #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < N; i++){
        sum += A[i] * A[i];
    }
    return sqrt(sum);
}


float matrix_multiply(double** res, double** A, double** B, int N, int M, int N1){
    // Matrices shapes: A = NxM, B = MxN1, res = NxN1
    float diff = 0; double old;
    #pragma omp parallel for reduction(max:diff) private(old)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N1; j++){
            old = res[i][j];
            res[i][j] = 0;
            for(int k = 0; k < M; k++)
                res[i][j] += A[i][k] * B[k][j];
            diff = std::max(diff, (float)fabs(fabs(res[i][j]) - fabs(old)));
        }
    }
    return diff;
}

void qr(double** Q, double** R, double** D, int N){
    for(int i = 0; i < N; i++){
        #pragma omp parallel for
        for(int j = 0; j < N; j++)
            Q[j][i] = D[j][i];

        #pragma omp parallel for
        for(int j = 0; j < i; j++)
            R[j][i] = dot(Q, D, j, i, N);
        
        for(int j = 0; j < i; j++){
            #pragma omp parallel for
            for(int p = 0; p < N; p++)
                Q[p][i] = Q[p][i] -  R[j][i] * Q[p][j];
        }
            
        R[i][i] = norm(Q, i, N);
        #pragma omp parallel for
        for(int j = 0; j < N; j++)
            Q[j][i] = Q[j][i]/R[i][i];
    }
}

void print_matrix(double** A, int M, int N, char* name){
    printf("\nMatrix %s\n", name);
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
    // Dt is D transpose = NxM
    double** Dt = empty_matrix(N, M);
    
    // DDt is D.Dt = MxM, so are Q and R
    double** DDt = empty_matrix(M, M);
    double** Q = empty_matrix(M, M);
    double** R = empty_matrix(M, M);
    
    // Compute Dt
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            Dt[j][i] = D[i*N + j];
        }
    }
    print_matrix(Dt, N, M, "Dt\0");

    // Multiply D.Dt = MxN . NxM = MxM
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++){
            for(int k = 0; k < N; k++)
                DDt[i][j] += D[i*N + k] * Dt[k][j];
        }
    }
    print_matrix(DDt, M, M, "DDt\0");

    // Get Eigenvalues of DDt i.e. Q and R
    double diff = 10;
    double** Di = empty_matrix(M, M);
    copy_matrix(Di, DDt, M, M);
    double** Ei = diagonal_matrix(M);
    double** Ei_temp = empty_matrix(M, M); int count = 0;
    while(diff >= 0.001){
        qr(Q, R, Di, M);
        diff = std::max(matrix_multiply(Di, R, Q, M, M, M),  matrix_multiply(Ei_temp, Ei, Q, M, M, M));
        copy_matrix(Ei, Ei_temp, M, M);
        printf("Diff = %f, %d\n", diff, count);count++;
        // print_matrix(Q, M, M, "Q\0");
        // print_matrix(R, M, M, "R\0");
        // print_matrix(Di, M, M, "Di\0");
        // print_matrix(Ei, M, M, "Ei\0");
    }

    // Extract eigenvalues into an array
    double* eigenvalues = new double[M];
    for(int i = 0; i < M; i++)
        eigenvalues[i] = fabs(Di[i][i]);
    
    std::sort(eigenvalues, eigenvalues + M);
    reverse_array(eigenvalues, M);

    for(int i = 0; i < M; i++)
        printf("Eigenvalue %d is %f\n", i, eigenvalues[i]);

    double** sigma = empty_matrix(N, M);
    double** sigma_inv = empty_matrix(M, N);
    for(int i = 0; i < N; i++){
        *(*SIGMA+i) = sqrt(eigenvalues[i]);
        sigma[i][i] = sqrt(eigenvalues[i]);
        sigma_inv[i][i] = (1.0 / sqrt(eigenvalues[i]));
        printf("Sigma %d = %f\n", i, *(*SIGMA+i));
    }

    double** Vt = empty_matrix(M, M);
    // Compute V_T
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++){
            *(*V_T + M*i + j) = Ei[j][i];
            Vt[i][j] = Ei[j][i];
        }
    }
    
    printf("Computed Vt\n");
    // Print V_T
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++){
            printf("%f ",*(*V_T + M*i + j));
        }
        printf("\n");
    }
    printf("Computed Sigma-1\n");
    // Print Sigma-1
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++){
            printf("%f ",sigma_inv[i][j]);
        }
        printf("\n");
    }

    double** temp = empty_matrix(N, N);
    double** temp2 = empty_matrix(N, N);
    double** U_temp = empty_matrix(N, N);
    matrix_multiply(temp, Dt, Ei, N, M, N);
    matrix_multiply(temp2, temp, sigma_inv, N, M, N);

    // Copy in U
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++){
            *(*U + N*i + j) = temp2[i][j]; 
            U_temp[i][j] = temp2[i][j];
        }

    printf("U\n");
    // Print U
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", *(*U + N*i + j));
        printf("\n");
    }

    matrix_multiply(temp, U_temp, sigma, N, N, M);
    matrix_multiply(temp2, temp, Vt, N, M, M);

    print_matrix(Dt, N, M, "Original Dt");
    print_matrix(temp2, N, M, "Final Dt");
    
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    printf("\n\nPCA M:%d N:%d\n", M, N);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", *(D + i*N + j));
        printf("\n");
    }

    double ret = double(retention)/100;
    double sumeigen = 0;
    for(int i = 0; i < N; i++){
        sumeigen += *(SIGMA + i);
        printf("Sigma %d is %f\n", i, *(SIGMA + i));
    }

    double sumret = 0; int k = 0;
    for(k = 0; k < N; k++){
        sumret += (*(SIGMA + k) / sumeigen);
        if(sumret >= ret)
            break;
    }

    *K = k+1;
    printf("K = %d\n", *K);
    double** W = empty_matrix(N, k+1);
    #pragma omp parallel
    for(int i = 0; i < N; i++){
        for(int j = 0; j <= k; j++){
            W[i][j] = *(U + N*i + j);
        }
    }

    // print_matrix(W, N, *K, "W\0");

    printf("D-Hat:\n");
    float* DHatTemp = (float *)malloc(sizeof(float)*((k+1) * M));
    for(int i = 0; i < M; i++){
        for(int j = 0; j <= k; j++){
            DHatTemp[i*(k+1) + j] = 0;
            for(int p = 0; p < N; p++){
                DHatTemp[i*(k+1) + j] += *(D + i*N + p) * W[p][j];
            }
            // printf("%f ", DHatTemp[i*N + j]);
        }
        // printf("\n");
    }

    D_HAT = &DHatTemp;
}
