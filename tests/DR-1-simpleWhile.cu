__global__ void test(float *A){
    int i = threadIdx.x;
    int x = 5;
    while(x > 0){
        A[i] = A[i+1];
        x--;
    }
}
