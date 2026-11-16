#include <cuda_runtime.h>
#include <cstdio>
__global__ void add(float* out) { out[threadIdx.x] = float(threadIdx.x) + 2.0f; }
int main() {
    int count=0; cudaError_t status=cudaGetDeviceCount(&count);
    if(status!=cudaSuccess || count<1) { std::fprintf(stderr,"CUDA device error: %s\n",cudaGetErrorString(status)); return 1; }
    float *device=nullptr, host[32];
    if(cudaMalloc(&device,sizeof(host))!=cudaSuccess) return 2;
    add<<<1,32>>>(device);
    status=cudaGetLastError();
    if(status!=cudaSuccess) return 3;
    status=cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost);
    cudaFree(device);
    if(status!=cudaSuccess) return 4;
    for(int i=0;i<32;i++) if(host[i]!=float(i)+2.0f) return 5;
    std::printf("CUDA toolkit kernel PASS; devices=%d; output[31]=%.1f\n",count,host[31]);
    return 0;
}
