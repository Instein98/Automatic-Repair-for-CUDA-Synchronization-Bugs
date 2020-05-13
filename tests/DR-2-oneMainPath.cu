__global__ void test(float *A, const int N){
    int i = threadIdx.x;
    float x = A[i+1];
    if (i < N){
        A[i] = x;  // DR
    }
}