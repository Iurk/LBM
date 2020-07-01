#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

#include <cuda.h>

#include "LBM.h"
#include "dados.h"

using namespace myGlobals;

__constant__ unsigned int q, Nx_d, Ny_d;
__constant__ double rho0_d, u_max_d, nu_d, tau_d;

__constant__ double cs_d, w0_d, ws_d, wd_d;
__device__ int *ex_d;
__device__ int *ey_d;

__device__ __forceinline__ size_t gpu_field0_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	return (Nx_d*(Ny_d*(d - 1) + y) + x);
}

__global__ void gpu_taylor_green(unsigned int, double*, double*, double*);
__global__ void gpu_init_equilibrium(double*, double*, double*, double*, double*);
__global__ void gpu_stream_collide_save(double *, double *, double*, double*, double*, double*, double*, double*, bool);
__global__ void gpu_compute_flow_properties(unsigned int, double*, double*, double*, double*);
__global__ void gpu_init_device_var(int *, int *);

__device__ void taylor_green_eval(unsigned int t, unsigned int x, unsigned int y, double *r, double *u, double *v){
	
	double kx = 2.0*M_PI/Nx_d;
	double ky = 2.0*M_PI/Ny_d;
	double td = 1.0/(nu_d*(kx*kx + ky*ky));

	double X = x + 0.5;
	double Y = y + 0.5;
	double ux = -u_max_d*sqrt(ky/kx)*cos(kx*X)*sin(ky*Y)*exp(-1.0*t/td);
	double uy =  u_max_d*sqrt(kx/ky)*sin(kx*X)*cos(ky*Y)*exp(-1.0*t/td);
	double P = -0.25*rho0_d*u_max_d*u_max_d*((ky/kx)*cos(2.0*kx*X) + (kx/ky)*cos(2.0*ky*Y))*exp(-2.0*t/td);
	double rho = rho0_d + 3.0*P;

	*r = rho;
	*u = ux;
	*v = uy;
}

__host__ void taylor_green(unsigned int t, double *r, double *u, double *v){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_taylor_green<<< grid, block >>>(t, r, u, v);
	getLastCudaError("gpu_taylor_green kernel error");
}

__global__ void gpu_taylor_green(unsigned int t, double *r, double *u, double *v){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	size_t sidx = gpu_scalar_index(x, y);
	taylor_green_eval(t, x, y, &r[sidx], &u[sidx], &v[sidx]);
}

__host__ void init_equilibrium(double *f0, double *f1, double *r, double *u, double *v){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_init_equilibrium<<< grid, block >>>(f0, f1, r, u, v);
	getLastCudaError("gpu_init_equilibrium kernel error");
}

__global__ void gpu_init_equilibrium(double *f0, double *f1, double *r, double *u, double *v){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	double rho = r[gpu_scalar_index(x, y)];
	double ux = u[gpu_scalar_index(x, y)];
	double uy = v[gpu_scalar_index(x, y)];

	double A = 1.0/(cs_d*cs_d);
	double B = 1.0/(2.0*cs_d*cs_d);

	double w0r = w0_d*rho;
	double wsr = ws_d*rho;
	double wdr = wd_d*rho;
	double omusq = 1.0 - B*(ux*ux + uy*uy);

	double Wrho[] = {w0r, wsr, wsr, wsr, wsr, wdr, wdr, wdr, wdr};

	f0[gpu_field0_index(x, y)] = Wrho[0]*(omusq);

	for(int n = 1; n < q; ++n){
		double eidotu = ux*ex_d[n] + uy*ey_d[n];
		f1[gpu_fieldn_index(x, y, n)] = Wrho[n]*(omusq + A*eidotu*(1.0 + B*eidotu));
	}
}

__host__ void stream_collide_save(double *f0, double *f1, double *f2, double *f0neq, double *f1neq, double *r, double *u, double *v, bool save){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_stream_collide_save<<< grid, block >>>(f0, f1, f2, f0neq, f1neq, r, u, v, save);
	getLastCudaError("gpu_stream_collide_save kernel error");
}

__global__ void gpu_stream_collide_save(double *f0, double *f1, double *f2, double *f0neq, double *f1neq, double *r, double *u, double *v, bool save){

	const double tauinv = 1.0/tau_d;
	const double omega = 1.0 - tauinv;     

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int xf1 = (x + 1)%Nx_d;		// Forward
	unsigned int yf1 = (y + 1)%Ny_d;		// Forward
	unsigned int xb1 = (Nx_d + x - 1)%Nx_d;	// Backward
	unsigned int yb1 = (Ny_d + y - 1)%Ny_d; // Backward

	double ft0 = f0[gpu_field0_index(x, y)];

	// Streaming step
	double ft1 = f1[gpu_fieldn_index(xb1, y, 1)];
	double ft2 = f1[gpu_fieldn_index(x, yb1, 2)];
	double ft3 = f1[gpu_fieldn_index(xf1, y, 3)];
	double ft4 = f1[gpu_fieldn_index(x, yf1, 4)];
	double ft5 = f1[gpu_fieldn_index(xb1, yb1, 5)];
	double ft6 = f1[gpu_fieldn_index(xf1, yb1, 6)];
	double ft7 = f1[gpu_fieldn_index(xf1, yf1, 7)];
	double ft8 = f1[gpu_fieldn_index(xb1, yf1, 8)];

	double f[] = {ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8};

	double rho = 0, ux_i = 0, uy_i = 0;

	for(int n = 0; n < q; ++n){
		rho += f[n];
		ux_i += f[n]*ex_d[n];
		uy_i += f[n]*ey_d[n];
	}

	double rhoinv = 1.0/rho;

	double ux = rhoinv*ux_i;
	double uy = rhoinv*uy_i;

	if(save){
		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux;
		v[gpu_scalar_index(x, y)] = uy;
	}

	double A = 1.0/(cs_d*cs_d);
	double B = 1.0/(2.0*cs_d*cs_d);

	double w0r = w0_d*rho;
	double wsr = ws_d*rho;
	double wdr = wd_d*rho;

	double tw0r = tauinv*w0_d*rho; 
    double twsr = tauinv*ws_d*rho; 
    double twdr = tauinv*wd_d*rho; 

    double W[] = {w0_d, ws_d, ws_d, ws_d, ws_d, wd_d, wd_d, wd_d, wd_d};
    double Wrho[] = {w0r, wsr, wsr, wsr, wsr, wdr, wdr, wdr, wdr};
    double tWrho[] = {tw0r, twsr, twsr, twsr, twsr, twdr, twdr, twdr, twdr};

	double omusq = 1.0 - B*(ux*ux + uy*uy);

	// Approximation of fneq
	f0neq[gpu_field0_index(x, y)] = f[0] - Wrho[0]*omusq;
	for(int n = 1; n < q; ++n){
		double eidotu = ux*ex_d[n] + uy*ey_d[n];
		f1neq[gpu_fieldn_index(x, y, n)] = f[n] - Wrho[n]*(omusq + A*eidotu*(1.0 + B*eidotu));
	}

	// Calculating the Viscous stress tensor
	double tauxx = 0, tauxy = 0, tauyy = 0;
	for(int n = 1; n < q; ++n){
		tauxx += f1neq[gpu_fieldn_index(x, y, n)]*ex_d[n]*ex_d[n];
		tauxy += f1neq[gpu_fieldn_index(x, y, n)]*ex_d[n]*ey_d[n];
		tauyy += f1neq[gpu_fieldn_index(x, y, n)]*ey_d[n]*ey_d[n];
	}

	f0[gpu_field0_index(x, y)] = omega*f0neq[gpu_field0_index(x, y)] + tWrho[0]*(omusq);

	for(int n = 1; n < q; ++n){
		f1neq[gpu_fieldn_index(x, y, n)] = B*W[n]*(tauxx*(A*ex_d[n] - 1.0) + 2.0*tauxy*A*ex_d[n]*ey_d[n] + tauyy*(A*ey_d[n] - 1.0));
		double eidotu = ux*ex_d[n] + uy*ey_d[n];
		f2[gpu_fieldn_index(x, y, n)] = omega*f1neq[gpu_fieldn_index(x, y, n)] + tWrho[n]*(omusq + A*eidotu*(1.0 + B*eidotu));
	}
}

__host__ void compute_flow_properties(unsigned int t, double *r, double *u, double *v, double *prop, double *prop_gpu, double *prop_host){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_compute_flow_properties<<< grid, block, 7*block.x*sizeof(double) >>>(t, r, u, v, prop_gpu);
	getLastCudaError("gpu_compute_flow_properties kernel error");

	size_t prop_size_bytes = 7*grid.x*grid.y*sizeof(double);
	checkCudaErrors(cudaMemcpy(prop_host, prop_gpu, prop_size_bytes, cudaMemcpyDeviceToHost));

	double E = 0.0;

	double sumrhoe2 = 0.0;
	double sumuxe2 = 0.0;
	double sumuye2 = 0.0;

	double sumrhoa2 = 0.0;
	double sumuxa2 = 0.0;
	double sumuya2 = 0.0;

	for(unsigned int i = 0; i < grid.x*grid.y; ++i){

		E += prop_host[7*i];
		sumrhoe2 += prop_host[7*i + 1];
		sumuxe2 += prop_host[7*i + 2];
		sumuye2 += prop_host[7*i + 3];

		sumrhoa2 += prop_host[7*i + 4];
		sumuxa2 += prop_host[7*i + 5];
		sumuya2 += prop_host[7*i + 6];
	}

	prop[0] = E;
	prop[1] = sqrt(sumrhoe2/sumrhoa2);
	prop[2] = sqrt(sumuxe2/sumuxa2);
	prop[3] = sqrt(sumuye2/sumuya2);
}

__global__ void gpu_compute_flow_properties(unsigned int t, double *r, double *u, double *v, double *prop_gpu){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	extern __shared__ double data[];

	double *E = data;
	double *rhoe2 = data + blockDim.x;
	double *uxe2 = data + 2*blockDim.x;
	double *uye2 = data + 3*blockDim.x;
	double *rhoa2 = data + 4*blockDim.x;
	double *uxa2 = data + 5*blockDim.x;
	double *uya2 = data + 6*blockDim.x;

	double rho = r[gpu_scalar_index(x, y)];
	double ux = u[gpu_scalar_index(x, y)];
	double uy = v[gpu_scalar_index(x, y)];

	E[threadIdx.x] = rho*(ux*ux + uy*uy);

	double rhoa, uxa, uya;
	taylor_green_eval(t, x, y, &rhoa, &uxa, &uya);

	rhoe2[threadIdx.x] = (rho - rhoa)*(rho - rhoa);
	uxe2[threadIdx.x] = (ux - uxa)*(ux - uxa);
	uye2[threadIdx.x] = (uy - uya)*(uy - uya);

	rhoa2[threadIdx.x] = (rhoa - rho0_d)*(rhoa - rho0_d);
	uxa2[threadIdx.x] = uxa*uxa;
	uya2[threadIdx.x] = uya*uya;

	__syncthreads();

	if (threadIdx.x == 0){
		
		size_t idx = 7*(gridDim.x*blockIdx.y + blockIdx.x);

		for(int n = 0; n < 7; ++n){
			prop_gpu[idx+n] = 0.0;
		}

		for(int i = 0; i < blockDim.x; ++i){
			prop_gpu[idx] += E[i];
			prop_gpu[idx+1] += rhoe2[i];
			prop_gpu[idx+2] += uxe2[i];
			prop_gpu[idx+3] += uye2[i];

			prop_gpu[idx+4] += rhoa2[i];
			prop_gpu[idx+5] += uxa2[i];
			prop_gpu[idx+6] += uya2[i];
		}
	}
}

__host__ void report_flow_properties(unsigned int t, double *rho, double *ux, double *uy,
									 double *prop_gpu, double *prop_host){

	double prop[4];
	compute_flow_properties(t, rho, ux, uy, prop, prop_gpu, prop_host);
	printf("%u, %g, %g, %g, %g\n", t, prop[0], prop[1], prop[2], prop[3]);
}

__host__ void save_scalar(const char* name, double *scalar_gpu, double *scalar_host, unsigned int n){

	char filename[128];
	char format[50];

	int ndigits = floor(log10((double)NSTEPS) + 1.0);

	sprintf(format, "%s%%s%%0%dd.bin", folder, ndigits);
	sprintf(filename, format, name, n);

	checkCudaErrors(cudaMemcpy(scalar_host, scalar_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));

	FILE* fout = fopen(filename, "wb+");

	fwrite(scalar_host, 1, mem_size_scalar, fout);

	if(ferror(fout)){
		fprintf(stderr, "Error saving to %s\n", filename);
		perror("");
	}
	else{
		if(!quiet){
			printf("Saved to %s\n", filename);
		}
	}

	fclose(fout);
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
	checkCudaErrors(cudaMemcpyToSymbol(ws_d, w_s, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(wd_d, w_d, sizeof(double)));
}

__host__ void init_device_var(){

	dim3 grid(1, 1, 1);
	dim3 block(1, 1, 1);

	int *temp_ex_d, *temp_ey_d;

	checkCudaErrors(cudaMalloc((void**)&temp_ex_d, ndir*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&temp_ey_d, ndir*sizeof(int)));

	checkCudaErrors(cudaMemcpy(temp_ex_d, ex, ndir*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(temp_ey_d, ey, ndir*sizeof(int), cudaMemcpyHostToDevice));

	gpu_init_device_var<<< grid, block >>>(temp_ex_d, temp_ey_d);
	getLastCudaError("gpu_init_device_var kernel error");

	checkCudaErrors(cudaFree(temp_ex_d));
	checkCudaErrors(cudaFree(temp_ey_d));
}

__global__ void gpu_init_device_var(int *temp_ex_d, int *temp_ey_d){
	ex_d = temp_ex_d;
	__syncthreads();
	ey_d = temp_ey_d;
}