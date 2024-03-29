#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

#define _USE_MATH_DEFINES
#include <math.h>

#include <cuda.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

#include "paths.h"
#include "LBM.h"
#include "dados.h"

using namespace myGlobals;

// Input data
__constant__ unsigned int q, Nx_d, Ny_d;
__constant__ double rho0_d, u_max_d, nu_d, tau_d;

//Lattice Data
__constant__ double cs_d, w0_d, wp_d, ws_d;
__device__ int *ex_d;
__device__ int *ey_d;

// Mesh data
__device__ bool *cylinder_d;
__device__ bool *fluid_d;

__device__ __forceinline__ size_t gpu_field0_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	return (Nx_d*(Ny_d*(d) + y) + x);
}

__global__ void gpu_init_equilibrium(double*, double*, double*, double*);
__global__ void gpu_stream_collide_save(double*, double*, double*, double*, double*, double*, double*, bool);
__global__ void gpu_compute_convergence(double*, double*, double*);
__global__ void gpu_compute_flow_properties(unsigned int, double*, double*, double*, double*);
__global__ void gpu_print_mesh(int);
__global__ void gpu_initialization(double*, double);

// Equilibrium
__device__ void gpu_equilibrium(unsigned int x, unsigned int y, double rho, double ux, double uy, double *feq){

	double cs2 = cs_d*cs_d;
	double cs4 = cs2*cs2;
	double cs6 = cs4*cs2;

	double A = 1.0/(cs2);
	double B = 1.0/(2.0*cs4);

	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d};
	for(int n = 0; n < q; ++n){
		
		double ux2 = ux*ux;
		double uy2 = uy*uy;
		double ex2 = ex_d[n]*ex_d[n];
		double ey2 = ey_d[n]*ey_d[n];

		double order_1 = A*(ux*ex_d[n] + uy*ey_d[n]);
		double order_2 = B*(ux2*(ex2 - cs2) + 2*ux*uy*ex_d[n]*ey_d[n] + uy2*(ey2 - cs2));

		feq[gpu_fieldn_index(x, y, n)] = W[n]*rho*(1 + order_1 + order_2);
	}
}

// Non Equilibrium
__device__ void gpu_nonequilibrium(unsigned int x, unsigned int y, double tauxx, double tauxy, double tauyy, double *fneq){

	double cs2 = cs_d*cs_d;
	double cs4 = cs2*cs2;

	double B = 1.0/(2.0*cs4);

	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d};
	for(int n = 0; n < q; ++n){
		
		double ex2 = ex_d[n]*ex_d[n];
		double ey2 = ey_d[n]*ey_d[n];

		double order_1 = B*(tauxx*(ex2 - cs2) + 2*tauxy*ex_d[n]*ey_d[n] + tauyy*(ey2 - cs2));

		fneq[gpu_fieldn_index(x, y, n)] = W[n]*(order_1);
	}	

}

__host__ void init_equilibrium(double *f1, double *r, double *u, double *v){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_init_equilibrium<<< grid, block >>>(f1, r, u, v);
	getLastCudaError("gpu_init_equilibrium kernel error");
}

__global__ void gpu_init_equilibrium(double *f1, double *r, double *u, double *v){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	double rho = r[gpu_scalar_index(x, y)];
	double ux = u[gpu_scalar_index(x, y)];
	double uy = v[gpu_scalar_index(x, y)];

	gpu_equilibrium(x, y, rho, ux, uy, f1);
}

__host__ void stream_collide_save(double *f1, double *f2, double *feq, double *fneq, double *r, double *u, double *v, bool save){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	//dim3 grid(1,1,1);
	//dim3 block(1,1,1);

	gpu_stream_collide_save<<< grid, block >>>(f1, f2, feq, fneq, r, u, v, save);
	getLastCudaError("gpu_stream_collide_save kernel error");
}

__global__ void gpu_stream_collide_save(double *f1, double *f2, double *feq, double *fneq, double *r, double *u, double *v, bool save){

	const double omega = 1.0/tau_d;

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int x_att, y_att;

	double rho = 0, ux_i = 0, uy_i = 0;
	for(int n = 0; n < q; ++n){
		rho += f1[gpu_fieldn_index(x, y, n)];
		ux_i += f1[gpu_fieldn_index(x, y, n)]*ex_d[n];
		uy_i += f1[gpu_fieldn_index(x, y, n)]*ey_d[n];
	}

	double ux = ux_i/rho;
	double uy = uy_i/rho;

	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;
	
	double cs2 = cs_d*cs_d;
	double cs4 = cs2*cs2;

	gpu_equilibrium(x, y, rho, ux, uy, feq);

	// Approximation of fneq
	for(int n = 0; n < q; ++n){
		fneq[gpu_fieldn_index(x, y, n)] = f1[gpu_fieldn_index(x, y, n)] - feq[gpu_fieldn_index(x, y, n)];
	}

	// Calculating the Viscous stress tensor
	double tauxx = 0, tauxy = 0, tauyy = 0;
	for(int n = 0; n < q; ++n){
		tauxx += fneq[gpu_fieldn_index(x, y, n)]*ex_d[n]*ex_d[n];
		tauxy += fneq[gpu_fieldn_index(x, y, n)]*ex_d[n]*ey_d[n];
		tauyy += fneq[gpu_fieldn_index(x, y, n)]*ey_d[n]*ey_d[n];
	}

	// Recalculating fneq
	gpu_nonequilibrium(x, y, tauxx, tauxy, tauyy, fneq);

	// Collision Step
	for(int n = 0; n < q; ++n){
		f1[gpu_fieldn_index(x, y, n)] = feq[gpu_fieldn_index(x, y, n)] + (1.0 - omega)*fneq[gpu_fieldn_index(x, y, n)];
	}

	// Stream Step
	for(int n = 0; n < q; ++n){
		x_att = (x + ex_d[n] + Nx_d)%Nx_d;
		y_att = (y + ey_d[n] + Ny_d)%Ny_d;

		f2[gpu_fieldn_index(x_att, y_att, n)] = f1[gpu_fieldn_index(x, y, n)];
	}
}

__host__ void report_flow_properties(unsigned int t, double *rho, double *ux, double *uy, double *prop_gpu, double *prop_host, bool msg){

	double prop[1];

	if(msg){
		compute_flow_properties(t, rho, ux, uy, prop, prop_gpu, prop_host);
		std::cout << std::setw(10) << t << std::setw(13) << prop[0] << std::setw(15) << std::endl;
	}
}

__host__ void compute_flow_properties(unsigned int t, double *r, double *u, double *v, double* prop, double *prop_gpu, double *prop_host){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_compute_flow_properties<<< grid, block, block.x*sizeof(double) >>>(t, r, u, v, prop_gpu);
	getLastCudaError("gpu_compute_flow_properties kernel error");

	size_t prop_size_bytes = grid.x*grid.y*sizeof(double);
	checkCudaErrors(cudaMemcpy(prop_host, prop_gpu, prop_size_bytes, cudaMemcpyDeviceToHost));

	double E = 0.0;

	for(unsigned int i = 0; i < grid.x*grid.y; ++i){
		E += prop_host[i];
	}

	prop[0] = E;
}

__global__ void gpu_compute_flow_properties(unsigned int t, double *r, double *u, double *v, double *prop_gpu){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	extern __shared__ double data[];

	double *E = data;

	double rho = r[gpu_scalar_index(x, y)];
	double ux = u[gpu_scalar_index(x, y)];
	double uy = v[gpu_scalar_index(x, y)];

	E[threadIdx.x] = rho*(ux*ux + uy*uy);

	__syncthreads();

	if (threadIdx.x == 0){
		
		size_t idx = 1*(gridDim.x*blockIdx.y + blockIdx.x);

		for(int n = 0; n < 1; ++n){
			prop_gpu[idx+n] = 0.0;
		}

		for(int i = 0; i < blockDim.x; ++i){
			prop_gpu[idx] += E[i];
		}
	}
}

void wrapper_input(unsigned int *nx, unsigned int *ny, double *rho, double *u, double *nu, const double *tau){
	checkCudaErrors(cudaMemcpyToSymbol(Nx_d, nx, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(Ny_d, ny, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(rho0_d, rho, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(u_max_d, u, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(nu_d, nu, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(tau_d, tau, sizeof(double)));
}

void wrapper_lattice(unsigned int *ndir, double *c, double *w_0, double *w_s, double *w_d){
	checkCudaErrors(cudaMemcpyToSymbol(q, ndir, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(cs_d, c, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(w0_d, w_0, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(wp_d, w_s, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(ws_d, w_d, sizeof(double)));
}

__host__ int* generate_e(int *e, std::string mode){

	int *temp_e;

	size_t mem_e = ndir*sizeof(int);

	checkCudaErrors(cudaMalloc(&temp_e, mem_e));
	checkCudaErrors(cudaMemcpy(temp_e, e, mem_e, cudaMemcpyHostToDevice));

	if(mode == "x"){
		checkCudaErrors(cudaMemcpyToSymbol(ex_d, &temp_e, sizeof(temp_e)));
	}
	else if(mode == "y"){
		checkCudaErrors(cudaMemcpyToSymbol(ey_d, &temp_e, sizeof(temp_e)));
	}

	return temp_e;
}

__host__ bool* generate_mesh(bool *mesh, std::string mode){

	int mode_num;
	bool *temp_mesh;

	checkCudaErrors(cudaMalloc(&temp_mesh, mem_mesh));
	checkCudaErrors(cudaMemcpy(temp_mesh, mesh, mem_mesh, cudaMemcpyHostToDevice));
	

	if(mode == "solid"){
		checkCudaErrors(cudaMemcpyToSymbol(cylinder_d, &temp_mesh, sizeof(temp_mesh)));
		mode_num = 1;
	}
	else if(mode == "fluid"){
		checkCudaErrors(cudaMemcpyToSymbol(fluid_d, &temp_mesh, sizeof(temp_mesh)));
		mode_num = 1;
	}

	if(meshprint){
		gpu_print_mesh<<< 1, 1 >>>(mode_num);
		printf("\n");
	}

	return temp_mesh;
}

__global__ void gpu_print_mesh(int mode){
	if(mode == 1){
		for(int y = 0; y < Ny_d; ++y){
			for(int x = 0; x < Nx_d; ++x){
				printf("%d ", cylinder_d[Nx_d*y + x]);
			}
		printf("\n");
		}
	}
	else if(mode == 2){
		for(int y = 0; y < Ny_d; ++y){
			for(int x = 0; x < Nx_d; ++x){
				printf("%d ", fluid_d[Nx_d*y + x]);
			}
		printf("\n");
		}
	}
}

__host__ void initialization(double *array, double value){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_initialization<<< grid, block >>>(array, value);
	getLastCudaError("gpu_print_array kernel error");
}

__global__ void gpu_initialization(double *array, double value){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	array[gpu_scalar_index(x, y)] = value;
}

__host__ bool* create_pinned_mesh(bool *array){

	bool *pinned;
	const unsigned int bytes = Nx*Ny*sizeof(bool);

	checkCudaErrors(cudaMallocHost((void**)&pinned, bytes));
	memcpy(pinned, array, bytes);
	return pinned;
}

__host__ double* create_pinned_double(){

	double *pinned;

	checkCudaErrors(cudaMallocHost((void**)&pinned, mem_size_scalar));
	return pinned;
}
