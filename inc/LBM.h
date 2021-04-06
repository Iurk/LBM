#ifndef __LBM_H
#define __LBM_H
#include <iostream>
#include <vector>

extern __constant__ unsigned int q, Nx_d, Ny_d;
extern __constant__ double rho0_d, u_max_d, nu_d;

extern __constant__ double cs_d, w0_d, wp_d, ws_p;

extern __device__ int *ex_d, ey_d*;
extern __device__ bool *cylinder, *fluid;

void initialization(double*, double);
void init_equilibrium(double*, double*, double*, double*);
void stream_collide_save(double*, double*, double*, double*, double*, double*, double*, bool);
void compute_flow_properties(unsigned int, double*, double*, double*, double*, double*, double*);
void report_flow_properties(unsigned int, double*, double*, double*, double*, double*);
void save_scalar(const std::string , double*, double*, unsigned int);
void wrapper_input(unsigned int*, unsigned int*, double*, double*, double*, const double*);
void wrapper_lattice(unsigned int*, double*, double*, double*, double*);
int* generate_e(int*, std::string);
bool* generate_mesh(bool*, std::string);
bool* create_pinned_mesh(bool *);
double* create_pinned_double();

#define checkCudaErrors(err) __checkCudaErrors(err, #err, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *const func, const char *const file, const int line){
	if(err != cudaSuccess){
		fprintf(stderr, "CUDA error at %s(%d)\"%s\": [%d] %s.\n", file, line, func, (int)err, cudaGetErrorString(err));
		exit(-1);
	}
}

inline void __getLastCudaError(const char *const errorMessage, const char *const file, const int line){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, "CUDA error at %s(%d): [%d] %s.\n", file, line, (int)err, cudaGetErrorString(err));
		exit(-1);
	}
}

#endif
