__global__ void test(float *A, const int N){
    int i = threadIdx.x;
    if (i < N){
        float x = A[i+1];
        A[i] = x;  // DR
    }
}