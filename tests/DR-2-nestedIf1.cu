__global__ void test(float *A, const int N){
    int i = threadIdx.x;
    if (i < 8){
        float x = A[i+1];
        if (i < N){
            A[i] = x;  // DR
        }
    }
}
