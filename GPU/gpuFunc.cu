#include "gpuHeaders.cuh" 
#include "math.h"

__global__ void calcRho(float *rho, float *f){
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int d = gridDim.y*blockDim.y;
    
    float sum = 0;
    for(int n=0; n < qd; n++){
        sum += f[(n*d + y)*w + x];
    }
    rho[y*w + x] = sum;
}

__global__ void calcU(float *rho, float *u, float *f){
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int d = gridDim.y*blockDim.y;
    
    float sumx = 0;
    float sumy = 0;
    for(int n=0; n < qd; n++){
        sumx += f[(n*d + y)*w + x]*exd[n];
        sumy += f[(n*d + y)*w + x]*eyd[n];
    }
    u[(0*2 + y)*w + x] = sumx/rho[y*w + x];
    u[(1*2 + y)*w + x] = sumy/rho[y*w + x];
}



__global__ void Equilibrium(float *rho, float *u, float *feq){
	
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int d = gridDim.y*blockDim.y;
    
    for(int n=0; n < qd; n++){
            float u_dot_e = u[(0*2 + y)*w + x]*exd[n] + u[(1*2 + y)*w + x]*eyd[n];
            float u_2 = pow(u[(0*2 + y)*w + x], 2) + pow(u[(1*2 + y)*w + x], 2);
            feq[(n*d + y)*w + x] = rho[y*w + x]*Wd[n]*(1.0 + Ad*u_dot_e + Bd*pow(u_dot_e, 2) - Cd*u_2);
            }
}