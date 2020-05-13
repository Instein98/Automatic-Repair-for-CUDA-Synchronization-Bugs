__global__ void test(float *A, const int N){
    int i = threadIdx.x;
    if (i < N){
        A[i] = A[i+1];
    }
}