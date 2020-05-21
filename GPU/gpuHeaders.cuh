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
__global__ void Propagation(float* f, float* fout);
__global__ void FluidWall(int* wall, int* solid);
__global__ void BounceBack(int* fluid, int* solid, float* f, float* fout);
__global__ void ZouHeIn(float u_inlet, float* rho, float* u, float* f);
__global__ void ZouHeOut(int Nx, float* rho, float* u, float* f);

#endif