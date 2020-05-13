__global__ void test(float *A){
    int i = threadIdx.x;
    if (i == 5){
        return;
    }
    int x = 2*4;
    int y = 1-2;
    A[i] = A[i+1];
}

__global__ void test1(float *A){
    int i = threadIdx.x;
    if (i != 5){
        A[i] = A[i+1];
    }else{
        return;
    }
}