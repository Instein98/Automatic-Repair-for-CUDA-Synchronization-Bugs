__global__ void test(float *A){
    int i = threadIdx.x;
    if (i % 2 == 0){
        float x = A[i+1];
    }else{
        A[i] = i;
    }
}