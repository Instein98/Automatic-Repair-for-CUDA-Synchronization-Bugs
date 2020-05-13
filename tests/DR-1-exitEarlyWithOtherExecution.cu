__global__ void test(float *A){
    int i = threadIdx.x;
    int x = 5;
    while(x > 0){
        A[i] = A[i+1];
        x--;
        if(i + x <= 3){
            int a = 5*4;
            break;  // or return
        }
    }
}

__global__ void test(float *A){
    bool earlyBreak = false;
    int i = threadIdx.x;
    int x = 5;
    while(x > 0){
        if(i + x > 3){
            A[i] = A[i+1];
            x--;
        }else{
            earlyBreak = true;
        }
    }
    if (earlyBreak){
        int a = 5*4;
    }
}
