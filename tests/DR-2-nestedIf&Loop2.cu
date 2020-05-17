__global__ void test(float *A, const int N){
    int i = threadIdx.x;
    if (i < N){
        int foo = 3 * 5 + 2;
        for(int j = 0; j < 10; j++){
            float x = A[i+1];
            A[i] = x;  // DR
        }
        int bar = A[N-1] + A[N-2];
    }else{
        int abc = 3 * 5 + 2;
        int bar = A[N-1] + A[N-2];
    }
}
