#include "gpuHeaders.cuh" 
#include "math.h"

__global__ void calcRho(float *rho, float *f){
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    float sum = 0;
    for(int n=0; n < qd; n++){
        sum += f[(n*h + y)*w + x];
    }
    rho[y*w + x] = sum;
}

__global__ void calcU(float *rho, float *u, float *f){
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    float sumx = 0;
    float sumy = 0;
    for(int n=0; n < qd; n++){
        sumx += f[(n*h + y)*w + x]*exd[n];
        sumy += f[(n*h + y)*w + x]*eyd[n];
    }
    u[(0*h + y)*w + x] = sumx/rho[y*w + x];
    u[(1*h + y)*w + x] = sumy/rho[y*w + x];
}

__global__ void Equilibrium(float *rho, float *u, float *feq){
	
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    for(int n=0; n < qd; n++){
        float u_dot_e = u[(0*h + y)*w + x]*exd[n] + u[(1*h + y)*w + x]*eyd[n];
        float u_2 = pow(u[(0*h + y)*w + x], 2) + pow(u[(1*h + y)*w + x], 2);
        feq[(n*h + y)*w + x] = rho[y*w + x]*Wd[n]*(1.0 + Ad*u_dot_e + Bd*pow(u_dot_e, 2) - Cd*u_2);
    }
}

__global__ void approxNonEquilibrium(float* f, float* feq, float* fneq){
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    for(int n=0; n < qd; n++){
        fneq[(n*h + y)*w + x] = f[(n*h + y)*w + x] - feq[(n*h + y)*w + x];
    }
}

__global__ void NonEquilibrium(float* tauab, float* fneq){
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    int Dtau = 2;
    
    for(int n=0; n < qd; n++){
        float tauxx = tauab[((0*Dtau + 0)*Dtau + y)*w + x]*(exd[n]*exd[n] - pow(csd, 2));
        float tauxy = tauab[((0*Dtau + 1)*Dtau + y)*w + x]*exd[n]*eyd[n];
        float tauyx = tauab[((1*Dtau + 0)*Dtau + y)*w + x]*exd[n]*eyd[n];
        float tauyy = tauab[((1*Dtau + 1)*Dtau + y)*w + x]*(eyd[n]*eyd[n] - pow(csd, 2));
        
        fneq[(n*h + y)*w + x] = Wd[n]*Bd*(tauxx + tauxy + tauyx + tauyy);
    }
}

__global__ void Tauab(float* tauab, float* fneq){
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    int Dtau = 2;
    float sum = 0;
    
    for(int a=0; a < Dtau; a++){
        for(int b=0; b < Dtau; b++){
            for(int n=0; n < qd; n++){
                sum += fneq[(n*h + y)*w + x]*exd[n]*eyd[n];
            }
            tauab[((a*Dtau + b)*Dtau + y)*w + x] = sum;
        }
    }
}

__global__ void Collision(float* fout, float* feq, float* fneq){
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    for(int n=0; n < qd; n++){
        fout[(n*h + y)*w + x] = feq[(n*h + y)*w + x] + (1 - omegad)*fneq[(n*h + y)*w + x];
    }
}