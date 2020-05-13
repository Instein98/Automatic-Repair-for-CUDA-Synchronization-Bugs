// #include <cuda_runtime.h>
#include <stdio.h>

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i< N; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match){
        printf("Arrays match!\n");
    }
}

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++){
        ip[i] = (float) (rand() & 0xFF)/10.0f;
    }
}

void printArr(float *A, const int N){
    for (int i = 0; i < N; i++){
        printf("%.2f ", A[i]);
    }
}

void moveArrayForwardHost(float *A, const int N){
    for (int idx = 0; idx < N; idx++){
        if (idx+1 < N){
            A[idx] = A[idx+1];
        }
    }
}

__global__ void moveArrayForwardDevice(float *A, const int N){
    int i = threadIdx.x;
    if (i+1 < N){
        A[i] = A[i+1];  // DR
    }
}

int main(){
    printf("Start..\n");

    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 32;  // Vector size

    size_t nBytes = nElem * sizeof(float);

    float *h_A, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    float *d_A;
    cudaMalloc((float**)&d_A, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    dim3 block (nElem);
    dim3 grid (nElem/block.x);

    moveArrayForwardDevice<<<grid, block>>>(d_A, nElem);
    cudaMemcpy(gpuRef, d_A, nBytes, cudaMemcpyDeviceToHost);

    moveArrayForwardHost(h_A, nElem);
    memcpy(hostRef, h_A, nBytes);

    printf("Result of Host: ");
    printArr(hostRef, nElem);
    printf("\n");
    printf("Result of Device: ");
    printArr(gpuRef, nElem);
    printf("\n");

    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_C);

    free(h_A);
    // free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
