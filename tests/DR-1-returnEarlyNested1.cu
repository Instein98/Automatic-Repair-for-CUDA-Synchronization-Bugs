// 核函数必须具有void返回类型！
// 提前退出与dr不发生在同一个一级对象中，都可以如此。
__global__ void test(float *A){
    int i = threadIdx.x;
    for(int j = 0; j < 8; j++){
        int a = 4 / 2;
        if (i % 2 == 0){
            int m = 5 * 10;
            return;
        }
    }
    A[i] = A[i+1];
}

//__global__ void test1(float *A){
//    bool shouldReturn = false;  // ++
//    int i = threadIdx.x;
//    for(int j = 0; j < 8; j++){
//        int a = 4 / 2;
//        if (a % 2 == 0){
//            int m = 5 * 10;
//            goto TARGET;
//        }
//    }
//    TARGET:
//    if (!shouldReturn){
//        A[i] = A[i+1];
//    }else{
//        return;
//    }
//}
//
//__global__ void test2(float *A){
//    bool shouldReturn = false;
//    float temp = 0;
//    int i = threadIdx.x;
//    for(int j = 0; j < 8; j++){
//        int a = 4 / 2;
//        if (a % 2 == 0){
//            int m = 5 * 10;
//            goto TARGET;
//        }
//    }
//    TARGET:
//    if (!shouldReturn){
//        temp = A[i+1];
//    }
//	__syncthreads();
//    if (!shouldReturn){
//	    A[i] = temp;
//    }else{
//        return;
//    }
//}
