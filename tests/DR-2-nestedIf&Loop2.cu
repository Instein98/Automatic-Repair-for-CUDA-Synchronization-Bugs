__global__ void test(float *A, const int N){
    int i = threadIdx.x;
    if (i < N){
        int abc = 3 * 5 + 2;
        for(int j = 0; j < 10; j++){
            float x = A[i+1];
            A[i] = x;  // DR
        }
        functionCall(abc);
    }else{
        int abc = 3 * 5 + 2;
        functionCall(abc);
    }
}
