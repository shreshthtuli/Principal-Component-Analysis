#include <malloc.h>
#include <omp.h>
#include <math.h>
#include <algorithm>

double compare_matrices(double** A, double** B, int M, int N){
    double diff = 0; int p, q;
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++){
            if(fabs(fabs(A[i][j]) - fabs(B[i][j])) > diff){
                diff = fabs(fabs(A[i][j]) - fabs(B[i][j]));
                p = i; q = j; 
            }
        }
    printf("i,j = %d,%d, %f\n", p, q, A[p][q]);
    return diff;
}

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
    double diff = 0; double old;
    #pragma omp parallel for reduction(max:diff) private(old)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N1; j++){
            old = res[i][j];
            res[i][j] = 0;
            for(int k = 0; k < M; k++)
                res[i][j] += A[i][k] * B[k][j];
            diff = std::max(diff, fabs(fabs(res[i][j]) - fabs(old)));
        }
    }
    return (float)diff;
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
    // Dc is copy of D = MxN
    double** Dc = empty_matrix(M, N);

    // DtD is Dt.D = NxN, so are Q and R
    double** DtD = empty_matrix(N, N);
    double** Q = empty_matrix(N, N);
    double** R = empty_matrix(N, N);
    
    // Compute Dt and Dc
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            Dt[j][i] = D[i*N + j];
            Dc[i][j] = D[i*N + j];
        }
    }

    // Multiply Dt.D = NxM . MxN = NxN
    matrix_multiply(DtD, Dt, Dc, N, M, N);

    // Get Eigenvalues of DtD i.e. Q and R
    double diff = 10, diff1, diff2;
    double** Di = empty_matrix(N, N);
    copy_matrix(Di, DtD, N, N);
    double** Ei = diagonal_matrix(N);
    double** Ei_temp = empty_matrix(N, N); int count = 0;
    while(diff >= 0.0001){
        qr(Q, R, Di, N);
        #pragma omp sections
        {
            #pragma omp section
                diff1 = matrix_multiply(Di, R, Q, N, N, N);
            #pragma omp section
                diff2 = matrix_multiply(Ei_temp, Ei, Q, N, N, N);
        }
        diff = std::max(diff1, diff2);
        copy_matrix(Ei, Ei_temp, N, N);
        // printf("Diff = %f, %d\n", diff, count);count++;
        // print_matrix(Q, N, N, "Q\0");
        // print_matrix(R, N, N, "R\0");
        // print_matrix(Di, N, N, "Di\0");
        // print_matrix(Ei, N, N, "Ei\0");
    }

    // Extract eigenvalues into an array
    double* eigenvalues = new double[N];
    for(int i = 0; i < N; i++)
        eigenvalues[i] = fabs(Di[i][i]);
    
    std::sort(eigenvalues, eigenvalues + N);
    reverse_array(eigenvalues, N);

    // for(int i = 0; i < N; i++)
    //     printf("Eigenvalue %d is %f\n", i, eigenvalues[i]);

    double** sigma = empty_matrix(M, N);
    double** sigma_inv = empty_matrix(N, M);
    for(int i = 0; i < N; i++){
        *(*SIGMA+i) = sqrt(eigenvalues[i]);
        sigma[i][i] = sqrt(eigenvalues[i]);
        sigma_inv[i][i] = (1.0 / sqrt(eigenvalues[i]));
        // printf("Sigma %d = %f\n", i, *(*SIGMA+i));
    }

    double** Vt = empty_matrix(M, M);
    double** U_temp = empty_matrix(N, N);
    // Compute U
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            *(*U + N*i + j) = Ei[i][j];
            U_temp[i][j] = Ei[i][j];
        }
    }
    
/*     printf("Computed U\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%f ",*(*U + N*i + j));
        }
        printf("\n");
    }
    printf("Computed Sigma-1\n");
    // Print Sigma-1
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            printf("%f ",sigma_inv[i][j]);
        }
        printf("\n");
    } */

    double** temp = empty_matrix(M, N);
    double** temp2 = empty_matrix(M, M);
    matrix_multiply(temp, Dc, Ei, M, N, N);
    matrix_multiply(temp2, temp, sigma_inv, M, N, M);

    // V_T
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < M; i++)
        for(int j = 0; j < M; j++){
            *(*V_T + M*j + i) = temp2[i][j]; 
            Vt[j][i] = temp2[i][j];
        }

/*     printf("Computed Vt\n");
    // Print V_T
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++)
            printf("%f ", *(*V_T + M*i + j));
        printf("\n");
    } */

    matrix_multiply(temp, U_temp, sigma, N, N, M);
    matrix_multiply(temp2, temp, Vt, N, M, M);

    print_matrix(Dt, N, M, "Original Dt");
    print_matrix(temp2, N, M, "Final Dt");
    printf("Comparison result diff = %f\n", compare_matrices(temp2, Dt, N, M));

    
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    printf("\n\nPCA M:%d N:%d\n", M, N);

    /* // Print D
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", *(D + i*N + j));
        printf("\n");
    } */

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

    // Print W
    // print_matrix(W, N, *K, "W\0");

    printf("D-Hat:\n");
    float* DHatTemp = (float *)malloc(sizeof(float)*((k+1) * M));
    for(int i = 0; i < M; i++){
        for(int j = 0; j <= k; j++){
            DHatTemp[i*(k+1) + j] = 0;
            for(int p = 0; p < N; p++){
                *(DHatTemp + i*(k+1) + j) += *(D + i*N + p) * W[p][j];
            }
            printf("%f ", DHatTemp[i*(k+1) + j]);
        }
        printf("\n");
    }

    D_HAT = &DHatTemp;
}
