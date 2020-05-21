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

__global__ void Propagation(float* f, float* fout){
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    if(y > 0){
        if(x > 0){
            f[(0*h + y)*w + x] = fout[(0*h + y)*w + x];
            f[(1*h + y)*w + x] = fout[(1*h + y)*w + (x-1)];
            f[(2*h + y)*w + x] = fout[(2*h + (y-1))*w + x];
            f[(3*h + y)*w + x] = fout[(3*h + y)*w + (x+1)];
            f[(4*h + y)*w + x] = fout[(4*h + (y+1))*w + x];
            f[(5*h + y)*w + x] = fout[(5*h + (y-1))*w + (x-1)];
            f[(6*h + y)*w + x] = fout[(6*h + (y-1))*w + (x+1)];
            f[(7*h + y)*w + x] = fout[(7*h + (y+1))*w + (x+1)];
            f[(8*h + y)*w + x] = fout[(8*h + (y+1))*w + (x-1)];
        }
    }
}

__global__ void FluidWall(int* fluid, int* solid){
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    
    if(solid[y*w + x] == 1){
        for(int n=0; n < qd; n++){
            int x_viz = x + exd[n];
            int y_viz = y + eyd[n];
            if(solid[y_viz*w + x_viz] == 0){
                fluid[y_viz*w + x_viz] = 1;
            }
        }
    }
}

__global__ void BounceBack(int* fluid, int* solid, float* f, float* fout){
    
    int noslip[] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    if(fluid[y*w + x] == 1){
        for(int n=0; n < qd; n++){
            int x_viz = x + exd[n];
            int y_viz = y + eyd[n];
            if(solid[y_viz*w + x_viz] == 1){
                int ns = noslip[n];
                f[(ns*h + y)*w + x] = fout[(n*h + y)*w + x];
            }
        }
    }
}

__global__ void ZouHeIn(float u_inlet, float* rho, float* u, float* f){

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    if(x == 0){
    
        int idxrho = y*w + x;
        int idxu = (0*h + y)*w + x;
        int idx0 = (0*h + y)*w + x;
        int idx2 = (2*h + y)*w + x;
        int idx3 = (3*h + y)*w + x;
        int idx4 = (4*h + y)*w + x;
        int idx6 = (6*h + y)*w + x;
        int idx7 = (7*h + y)*w + x;
        
        int idx1 = (1*h + y)*w + x;
        int idx5 = (5*h + y)*w + x;
        int idx8 = (8*h + y)*w + x;
        
        u[idxu] = u_inlet;        
        rho[idxrho] = (f[idx0] + f[idx2] + f[idx4] + 2*(f[idx3] + f[idx6] + f[idx7]))/(1 - u[idxu]);
        
        f[idx1] = f[idx3] + (2/3)*rho[idxrho]*u[idxu];
        f[idx5] = f[idx7] - (1/2)*(f[idx2] - f[idx4]) + (1/6)*rho[idxrho]*u[idxu];
        f[idx8] = f[idx6] + (1/2)*(f[idx2] - f[idx4]) + (1/6)*rho[idxrho]*u[idxu];
    }
}

__global__ void ZouHeOut(int Nx, float* rho, float* u, float* f){

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    if(x == Nx){
    
        int idxrho = y*w + x;
        int idxu = (0*h + y)*w + x;
        int idx0 = (0*h + y)*w + x;
        int idx1 = (1*h + y)*w + x;
        int idx2 = (2*h + y)*w + x;
        int idx4 = (4*h + y)*w + x;
        int idx5 = (5*h + y)*w + x;
        int idx8 = (8*h + y)*w + x;
        
        int idx3 = (3*h + y)*w + x;
        int idx6 = (6*h + y)*w + x;
        int idx7 = (7*h + y)*w + x;
        
        rho[idxrho] = 1.0;
        u[idxu] = (f[idx0] + f[idx2] + f[idx4] + 2*(f[idx1] + f[idx5] + f[idx8]))/rho[idxrho] - 1;        
        
        f[idx3] = f[idx1] - (2/3)*rho[idxrho]*u[idxu];
        f[idx6] = f[idx8] - (1/2)*(f[idx2] - f[idx4]) - (1/6)*rho[idxrho]*u[idxu];
        f[idx7] = f[idx5] + (1/2)*(f[idx2] - f[idx4]) - (1/6)*rho[idxrho]*u[idxu];
    }
}