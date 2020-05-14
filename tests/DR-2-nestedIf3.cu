__global__ void test(float *A, const int N){
    int i = threadIdx.x;
    float x = 0;
    if (i < 8){
        if (i < N){
            x = A[i+1];
        }else{
            if (i % 2 == 1){
                if (i < 4){
                    A[i] = x;
                }
            }
        }
    }
}
