__global__ void test(float *A){
    int i = threadIdx.x;
    A[i] = A[i+1];
}