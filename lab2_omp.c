#include <malloc.h>
#include <omp.h>
#include <math.h>
#include <algorithm>

void reverse_array(float* a, int N){
    float* temp = new float[N];
    for(int i = 0; i < N; i++)
        temp[i] = a[i];
    for(int i = 0; i < N; i++)
        a[i] = temp[N-i-1];
}

float** empty_matrix(int m, int n){
    float** A = new float*[m];
    for(int i = 0; i < m; i++){
        A[i] = new float[n];
        for(int j = 0; j < n; j++)  
            A[i][j] = 0;
    }
    return A;
}

void copy_matrix(float** to, float** from, int n, int m){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++)  
            to[i][j] = from[i][j];
    }
}

float** diagonal_matrix(int n){
    float** A = new float*[n];
    for(int i = 0; i < n; i++){
        A[i] = new float[n];
        for(int j = 0; j < n; j++)  
            A[i][j] = (i == j) ? 1 : 0;
    }
    return A;
}

float dot(float** A, float** B, int cola, int colb, int N){
    float sum = 0;
    for(int i = 0; i < N; i++){
        sum += A[i][cola] * B[i][colb];
    }
    return sum;
}

float norm(float** A, int cola, int N){
    float sum = 0;
    for(int i = 0; i < N; i++){
        sum += A[i][cola] * A[i][cola];
    }
    return sqrt(sum);
}

float norm_vector(float* A, int N){
    float sum = 0;
    for(int i = 0; i < N; i++){
        sum += A[i] * A[i];
    }
    return sqrt(sum);
}


float matrix_multiply(float** res, float** A, float** B, int N, int M, int N1){
    // Matrices shapes: A = NxM, B = MxN1, res = NxN1
    float diff = 0; float old;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N1; j++){
            old = res[i][j];
            res[i][j] = 0;
            for(int k = 0; k < M; k++)
                res[i][j] += A[i][k] * B[k][j];
            diff += fabs(fabs(res[i][j]) - fabs(old));
        }
    }
    return diff;
}

void qr(float** Q, float** R, float** D, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            Q[j][i] = D[j][i];

        for(int j = 0; j < i; j++){
            R[j][i] = dot(Q, D, j, i, N);
            for(int p = 0; p < N; p++)
                Q[p][i] = Q[p][i] - R[j][i] * Q[p][j];
        }
            
        R[i][i] = norm(Q, i, N);
        for(int j = 0; j < N; j++)
            Q[j][i] = Q[j][i]/R[i][i];
    }
}

void print_matrix(float** A, int M, int N, char* name){
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
    float** Dt = empty_matrix(N, M);
    
    // DDt is D.Dt = MxM, so are Q and R
    float** DDt = empty_matrix(M, M);
    float** Q = empty_matrix(M, M);
    float** R = empty_matrix(M, M);
    
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
    float diff = 10;
    float** Di = empty_matrix(M, M);
    copy_matrix(Di, DDt, M, M);
    float** Ei = diagonal_matrix(M);
    float** Ei_temp = empty_matrix(M, M);
    while(diff > 1.5){
        diff = 0;
        qr(Q, R, Di, M);
        diff += matrix_multiply(Di, R, Q, M, M, M);
        diff += matrix_multiply(Ei_temp, Ei, Q, M, M, M);
        copy_matrix(Ei, Ei_temp, M, M);
        printf("Diff = %f\n", diff);
        // print_matrix(Q, M, M, "Q\0");
        // print_matrix(R, M, M, "R\0");
        // print_matrix(Di, M, M, "Di\0");
        // print_matrix(Ei, M, M, "Ei\0");
    }

    // Extract eigenvalues into an array
    float* eigenvalues = new float[M];
    for(int i = 0; i < M; i++)
        eigenvalues[i] = fabs(Di[i][i]);
    
    std::sort(eigenvalues, eigenvalues + M);
    reverse_array(eigenvalues, M);

    for(int i = 0; i < M; i++)
        printf("Eigenvalue %d is %f\n", i, eigenvalues[i]);

    float** sigma = empty_matrix(N, M);
    float** sigma_inv = empty_matrix(M, N);
    for(int i = 0; i < N; i++){
        *(*SIGMA+i) = sqrt(eigenvalues[i]);
        sigma_inv[i][i] = (1.0 / sqrt(eigenvalues[i]));
        printf("Sigma %d = %f\n", i, *(*SIGMA+i));
    }

    // Computer V_T
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++){
            *(*V_T + M*i + j) = Ei[j][i];
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

    float** temp = empty_matrix(N, N);
    float** temp2 = empty_matrix(N, N);
    matrix_multiply(temp, Dt, Ei, N, M, N);
    matrix_multiply(temp2, temp, sigma_inv, N, M, N);

    // Copy in U
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            *(*U + N*i + j) = temp2[i][j]; 

    printf("U\n");
    // Print U
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%f ", *(*U + N*i + j));
        printf("\n");
    }
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    float ret = float(retention)/100;
    float sumeigen = 0;
    for(int i = 0; i < N; i++){
        sumeigen += *(SIGMA + i);
        printf("Sigma %d is %f\n", i, *(SIGMA + i));
    }

    float sumret = 0; int k = 0;
    for(k = 0; k < N; k++){
        sumret += (*(SIGMA + k) / sumeigen);
        if(sumret >= ret)
            break;
    }

    *K = k+1;
    printf("K = %d\n", *K);
    float** W = empty_matrix(N, k+1);
    for(int i = 0; i < N; i++){
        for(int j = 0; j <= k; j++){
            W[i][j] = *(U + N*i + j);
        }
    }

    print_matrix(W, N, *K, "W\0");

    printf("D-Hat:\n");
    float* DHatTemp = (float *)malloc(sizeof(float)*((k+1) * M));
    for(int i = 0; i < M; i++){
        for(int j = 0; j <= k; j++){
            DHatTemp[i*(k+1) + j] = 0;
            for(int p = 0; p < N; p++){
                DHatTemp[i*N + j] += *(D + i*N + p) * W[p][j];
            }
            printf("%f ", DHatTemp[i*N + j]);
        }
        printf("\n");
    }

    D_HAT = &DHatTemp;
}
