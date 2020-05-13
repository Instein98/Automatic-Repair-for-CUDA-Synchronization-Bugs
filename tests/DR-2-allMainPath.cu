__global__ void test(float *A){
    int i = threadIdx.x;
    float x = A[i+1];
    A[i] = x;  // DR
}