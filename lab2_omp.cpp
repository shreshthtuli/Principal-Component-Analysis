#include <malloc.h>
#include <omp.h>


// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
    // Dt is D transpose = NxM
    float** Dt = new float*[N];
    for(int i = 0; i < N; i++)
        Dt[i] = new float[M];
    
    // DtD is Dt.D = NxN
    float** DtD = new float*[N];
    for(int i = 0; i < N; i++)
        DtD[i] = new float[N];
    
    // Compute Dt
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            Dt[j][i] = D[i*N + j];
        }
    }

    // Multiply Dt.D = NxM . MxN = NxN
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            DtD[i][j] = 0;
            for(int k = 0; k < M; k++)
                DtD[i][j] += Dt[i][k] * D[k*N + j];
        }
    }

    // Get Eigenvalues of DtD


}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    
}
