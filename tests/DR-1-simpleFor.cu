__global__ void test(float *A){
    int i = threadIdx.x;
    for(int j = 0; j < 5; j++){
        A[i] = A[i+1];
    }
}
