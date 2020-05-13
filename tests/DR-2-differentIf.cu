__global__ void test(float *A){
    int i = threadIdx.x;
    if (i % 2 == 0){
        float x = A[i];
    }
    if (i % 3 == 0){
        A[i] = i;
    }
}