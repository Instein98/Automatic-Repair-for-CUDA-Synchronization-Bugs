__global__ void test(float *A, const int N){
    int i = threadIdx.x;
    if (i < 8){
        float x = 0;
        if (i < N){
            x = A[i+1];
        }
        if (i % 2 == 1){
            if (i < 4){
                A[i] = x;
            }
        }
    }
}
