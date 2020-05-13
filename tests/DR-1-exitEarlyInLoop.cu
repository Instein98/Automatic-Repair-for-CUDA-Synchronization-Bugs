// 提前退出与DR发生在同一个一级对象中，并且它们的父辈对象有循环

__global__ void test(float *A){
    int i = threadIdx.x;
    int x = 5;
    while(x > 0){
        A[i] = A[i+1];
        x--;
        if(i + x <= 3)
            break;  // or return
    }
}
