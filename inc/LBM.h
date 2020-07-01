#ifndef __LBM_H
#define __LBM_H
#include <vector>

void init_equilibrium(double *, double *, double *, double *, double *);
void stream_collide_save(double *, double *, double *, double *, double *, double *, double *, double *, bool);
void compute_flow_properties(unsigned int, double *, double *, double *, double *, double *, double *);
void report_flow_properties(unsigned int, double *, double *, double *, double *, double *);
void save_scalar(const char* , double *, double *, unsigned int);
void wrapper_input(unsigned int *, unsigned int *, double *, double *, double *, const double *);
void wrapper_lattice(unsigned int *, double *, double *, double *, double *);
void init_device_var();
void generate_mesh(bool *);

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
