__global__ void test(float *A, const int N){
    int i = threadIdx.x;
    if (i - 3 != 0){
        for(int m = 0; m < 10; m++){
            if (i < N){
                int abc = 3 * 5 + 2;
                int bar = 3 * 5 + 2;
            }else{
                int abc = 3 * 5 + 2;
                for(int j = 0; j < 10; j++){
                    float x = A[i+1];
                    A[i] = x;  // DR
                }
                int foo = 3 * 5 + 2;
            }
        }
    }
}

//__global__ void test(float *A, const int N){
//    int i = threadIdx.x;
//    int p = 5;
//    if (i - 3 != 0){
//        while (p > 0){
//            if (i < N){
//                int abc = 3 * 5 + 2;
//                int bar = 3 * 5 + 2;
//            }else{
//                int abc = 3 * 5 + 2;
//                for(int j = 0; j < 10; j++){
//                    float x = A[i+1];
//                    A[i] = x;  // DR
//                }
//                int foo = 3 * 5 + 2;
//            }
//            p--;  // can not fix here
//        }
//    }
//}
