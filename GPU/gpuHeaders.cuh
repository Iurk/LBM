#ifndef CUDA_INCLUDES
#define CUDA_INCLUDES

__device__ __constant__ int qd;
__device__ __constant__ int exd[9];
__device__ __constant__ int eyd[9];

__device__ __constant__ float csd;
__device__ __constant__ float Wd[9];
__device__ __constant__ float Ad;
__device__ __constant__ float Bd;
__device__ __constant__ float Cd;
__device__ __constant__ float omegad;

__global__ void calcRho(float *rho, float *f);
__global__ void calcU(float *rho, float *u, float *f);
__global__ void Equilibrium(float *rho, float *u, float *feq);
__global__ void approxNonEquilibrium(float* f, float* feq, float* fneq);
__global__ void NonEquilibrium(float* tauab, float* fneq);
__global__ void Tauab(float* tauab, float* fneq);
__global__ void Collision(float* fout, float* feq, float* fneq);

#endif