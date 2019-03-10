#include <malloc.h>
#include <omp.h>
#include <math.h>
#include <algorithm>

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
        sum += A[cola][i] * B[colb][i];
    }
    return sum;
}

float norm(float** A, int cola, int N){
    float sum = 0;
    for(int i = 0; i < N; i++){
        sum += A[cola][i] * A[cola][i];
    }
    return sqrt(sum);
}

float* getvi(int i, float** Q, float** A, int N){
    float* res = new float[N];
    for(int j = 0; j < N; j++)
        res[j] = A[i][j];

    float dotp = 0;
    for(int j = 0; j < i; j++){
        dotp = dot(A, Q, i, j, N);
        for(int k = 0; k < N; k++){
            res[k] -= dotp * Q[j][k];
        }
    }
    return res;
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
            diff += fabs(res[i][j] - old);
        }
    }
    return diff;
}

void qr(float** Q, float** R, float** D, int N){
    for(int i = 0; i < N; i++){
        float normi = norm(D, i, N);
        float* vi = getvi(i, Q, D, N);

        for(int j = 0; j < N; j++)
            Q[i][j] = vi[j] / normi;
            
        for(int j = 0; j < N; j++)
            R[i][j] = (j < i) ? 0 : dot(D, Q, j, i, N); 
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
    
    // DtD is Dt.D = NxN, so are Q and R
    float** DtD = empty_matrix(N, N);
    float** Q = empty_matrix(N, N);
    float** R = empty_matrix(N, N);
    
    // Compute Dt
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            Dt[j][i] = D[i*N + j];
        }
    }

    // Multiply Dt.D = NxM . MxN = NxN
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < M; k++)
                DtD[i][j] += Dt[i][k] * D[k*N + j];
        }
    }

    // Get Eigenvalues of DtD i.e. Q and R
    float diff = 10;
    float** Di = empty_matrix(N, N);
    copy_matrix(Di, DtD, N, N);
    float** Ei = diagonal_matrix(N);
    while(diff > 1){
        diff = 0;
        qr(Q, R, Di, N);
        diff += matrix_multiply(Di, R, Q, N, N, N);
        diff += matrix_multiply(Ei, Ei, Q, N, N, N);
    }

    // Extract eigenvalues into an array
    float* eigenvalues = new float[N];
    for(int i = 0; i < N; i++)
        eigenvalues[i] = Di[i][i];
    
    std::sort(eigenvalues, eigenvalues + N);
    std::reverse(eigenvalues + N, eigenvalues);

    float** sigma = empty_matrix(N, N);
    float** sigma_inv = empty_matrix(N, N);
    for(int i = 0; i < N; i++){
        SIGMA[i][i] = sqrt(eigenvalues[i]);
        sigma_inv[i][i] = (1 / SIGMA[i][i]);
    }

    qr(Q, R, DtD, N);
    // Computer V_T
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            V_T[i][j] = R[j][i];
        }
    }
    
    float** temp = empty_matrix(M, N);
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++)
                temp[i][j] += D[M*i + k] * R[k][j];
        }
    }
    matrix_multiply(temp, temp, sigma_inv, M, N, N);
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
        sumeigen += SIGMA[i*N +i];
    }

    float sumret = 0; int k = 0;
    for(k = 0; k < N; k++){
        sumret += (SIGMA[k*N + k] / sumeigen);
        if(sumret >= ret)
            break;
    }

    *K = k+1;
    float** W = empty_matrix(N, k+1);
    for(int i = 0; i <= k; i++){
        for(int j = 0; j < N; j++){
            W[i][j] = U[i*N + j];
        }
    }

    float* DHatTemp = (float *)malloc(sizeof(float)*((k+1) * M));
    for(int i = 0; i < M; i++){
        for(int j = 0; j < k+1; j++){
            DHatTemp[i*N + j] = 0;
            for(int p = 0; p < N; p++){
                DHatTemp[i*N + j] += D[i*N + p] * W[p][j];
            }
        }
    }

    D_HAT = &DHatTemp;
}
