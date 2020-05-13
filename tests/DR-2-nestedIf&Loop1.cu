__global__ void test(float *A, const int N){
    int i = threadIdx.x;
    if (i < N){
        for(int j = 0; j < 10; j++){
            float x = A[i+1];
            A[i] = x;  // DR
        }
    }
}
