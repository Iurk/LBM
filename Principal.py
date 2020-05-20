#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:19:27 2019

@author: iurk
"""
import numpy as np
from time import time
import pycuda.autoinit
import funcoes_dados as fd
from Lattice import Lattice
import pycuda.driver as cuda
from Simulation import Simulation
from pycuda.compiler import SourceModule

def get_grid_block(Nx, Ny, block_x=1, block_y=1, block_z=1):
    block = [block_x, block_y, block_z]
    
    if block_x != 1:
        block[0] = block_x
    if block_y != 1:
        block[1] = block_y
    if block_z != 1:
        block[2] = block_z
        
    block = tuple(block)
    
    gpu = cuda.Device(0)
    Max_thread_per_block = gpu.max_threads_per_block
    print("Verifing...")
    if (block_x*block_y*block_z <= Max_thread_per_block):
        print("Max Thread per block: {}".format(Max_thread_per_block))
        print("Block size: {}".format(block_x*block_y*block_z))
        print("Ok!")
        grid_x = int(np.ceil(Nx/block[0]))
        grid_y = int(np.ceil(Ny/block[1]))
        grid = (grid_x, grid_y, 1)
        print("Problem Dim: {}x{}; Block Dim: {}x{}; Grid Dim:{}x{}".format(Nx, Ny, block_x, block_y, grid_x, grid_y))
        return grid, block
        
    else:
        print("Max Thread per block: {}".format(Max_thread_per_block))
        print("Block size: {}".format(block_x*block_y*block_z))
        print("Not Ok!")
        
#***** CUDA Code *****
#***** CUDA Files *****
headerFile = open('./GPU/gpuHeaders.cuh')
functionsFile = open('./GPU/gpuFunc.cu')

#***** Compilation *****
mod = SourceModule(headerFile.read() + functionsFile.read())
                                  
#***** Start Code *****   
ini = time()

#***** Input Data *****
Nx = 256                   # Número de partículas em x [Lattice units]
Ny = 128                    # Número de partículas em y [Lattice units]

Cx = Nx/4                   # Centro do Cilindro em x [Lattice units]
Cy = Ny/2                   # Centro do Cilindro em y [Lattice units]
D = 128                     # Diâmetro do Cilindro [Lattice units]

#***** Air Properties *****
rho_ar = 1.0                #1.21
mi_ar = 1.81e-5

#***** Flux Properties *****
uini = 0.04
Reynolds = [300]            # Reynolds Numbers
cl_Re = []
cd_Re = []

maxiter = 5000              # Número de Iterações

#***** CUDA Data *****
#***** Block Dim *****
block_x = 32
block_y = 16
#***** Events *****
start = cuda.Event()
stop = cuda.Event()

#***** Creating Lattice *****
lattice = 'D2Q9'
D2Q9 = Lattice(lattice)
q, cs, W, ex, ey = D2Q9.get_attributes()
A, B, C = D2Q9.get_param_dist_eq()

#***** Setting up the Simulation *****
simulation = Simulation(Nx, Ny, Cx, Cy, D, uini)

# #***** Setting up the GPU grid and blocks *****
grid, block = get_grid_block(Nx, Ny, block_x, block_y)

#***** Defining the cylinder *****
simulation.solid_create()
cilindro = simulation.get_solid()

#***** Initializing rho and u matrix *****
rho, u = simulation.initialize()

feq = np.zeros((q, Nx, Ny), dtype=np.float32)

#***** Calculation array's size *****
sizef = int(4*q*Nx*Ny)
sizetau = int(4*2*2*Nx*Ny)

#***** Device Memory Allocation *****
#***** Registers for Pinned Allocation *****
rho = cuda.register_host_memory(rho)
u = cuda.register_host_memory(u)
feq = cuda.register_host_memory(feq)

#***** Memory Allocation *****
rho_d = cuda.mem_alloc(rho.nbytes)
u_d = cuda.mem_alloc(u.nbytes)
f_d = cuda.mem_alloc(sizef)
feq_d = cuda.mem_alloc(sizef)
fneq_d = cuda.mem_alloc(sizef)
fout_d = cuda.mem_alloc(sizef)
tauab_d = cuda.mem_alloc(sizetau)

#***** Global Memory Allocation *****
qd = mod.get_global('qd')
csd = mod.get_global('csd')
Wd = mod.get_global('Wd')
exd = mod.get_global('exd')
eyd = mod.get_global('eyd')
Ad = mod.get_global('Ad')
Bd = mod.get_global('Bd')
Cd = mod.get_global('Cd')
omegad = mod.get_global('omegad')

#***** Sending Data to Device *****
#***** Global Data *****
cuda.memcpy_htod(qd[0], q)
cuda.memcpy_htod(csd[0], cs)
cuda.memcpy_htod(Wd[0], W)
cuda.memcpy_htod(exd[0], ex)
cuda.memcpy_htod(eyd[0], ey)
cuda.memcpy_htod(Ad[0], A)
cuda.memcpy_htod(Bd[0], B)
cuda.memcpy_htod(Cd[0], C)

#***** Normal Data *****
cuda.memcpy_htod_async(rho_d, rho)
cuda.memcpy_htod_async(u_d, u)

#***** Getting CUDA Functions *****
Rho = mod.get_function('calcRho')
Uest = mod.get_function('calcU')
Equilibrium = mod.get_function('Equilibrium')
approxNonEquilibrium = mod.get_function('approxNonEquilibrium')
NonEquilibrium = mod.get_function('NonEquilibrium')
Tauab = mod.get_function('Tauab')
Collision = mod.get_function('Collision')

#***** Preparing Functions *****
Rho.prepare('PP')
Uest.prepare('PPP')
Equilibrium.prepare('PPP')
approxNonEquilibrium.prepare('PPP')
NonEquilibrium.prepare('PP')
Tauab.prepare('PP')
Collision.prepare('PPP')

f = np.empty_like(feq)
fneq = np.empty_like(feq)
fout = np.empty_like(feq)
tauab = np.zeros((2, 2, Nx, Ny), dtype=np.float32)

#***** Reynolds Loop *****
for Re in Reynolds:
    cl_step = []
    cd_step = []
    
    #***** Creating Data Folders *****
    print('\nRe = {}'.format(Re))
    folder, folder_vel, folder_pres = fd.criar_pasta(Re)
    
    #***** Calculating Relaxation Term *****
    simulation.relaxation_term(cs, Re)
    omega = simulation.omega
    
    #***** Sendig omega to device *****
    cuda.memcpy_htod(omegad[0], omega)
    
    print('Runing...')
    
    print('Initializing...')
    Equilibrium.prepared_call(grid, block, rho_d, u_d, feq_d)
    cuda.memcpy_dtod_async(f_d, feq_d, sizef)
    
    cuda.memcpy_dtoh_async(f, f_d)
    cuda.memcpy_dtoh_async(feq, feq_d)
    
    #***** Main Loop *****
    step = 0
    while True:
        #***** Updating Momentum *****
        Rho.prepared_call(grid, block, rho_d, f_d)
        Uest.prepared_call(grid, block, rho_d, u_d, f_d)
        
        #***** Solving LBM BGK Regularized *****
        Equilibrium.prepared_call(grid, block, rho_d, u_d, feq_d)
        approxNonEquilibrium.prepared_call(grid, block, f_d, feq_d, fneq_d)
        Tauab.prepared_call(grid, block, tauab_d, fneq_d)
        NonEquilibrium.prepared_call(grid, block, tauab_d, fneq_d)
        Collision.prepared_call(grid, block, fout_d, feq_d, fneq_d)
        
        #***** Propagation Step *****
        

        
        cuda.memcpy_dtoh_async(rho, rho_d)
        cuda.memcpy_dtoh_async(u, u_d)
        cuda.memcpy_dtoh_async(f, f_d)
        cuda.memcpy_dtoh_async(feq, feq_d)
        cuda.memcpy_dtoh_async(fneq, fneq_d)
        cuda.memcpy_dtoh_async(fout, fout_d)
        cuda.memcpy_dtoh_async(tauab, tauab_d)
        
        print()




#***** Sending Data to Host *****
   

'''
#***** Construção do Cilindro e Perfil de Velocidades *****
solido = LBM.cilindro(Nx, Ny, Cx, Cy, D_est)

for Re in Reynolds:
    ini = time()
    cl_step = []
    cd_step = []
    
    print('\nRe = {}'.format(Re))
    folder, folder_vel, folder_pres = fd.criar_pasta(Re)
    tau, omega = LBM.relaxation_parameter(L, H, Nx, Ny, D_est, cs, uini, Re, mode)
    
    u_inlet = LBM.velocidade_lattice_units(Nx, Ny, uini, mode)
    
#***** Inicialização *****
    print('Initializing')
    feq = LBM.dist_eq(Nx, Ny, u_inlet, e, cs, n, 1.0, W)
    f = feq.copy()
    
    print('Running...')
    step = 0
    while True:
        rho = LBM.rho(f)
        u = cp.dot(e.transpose(), f.transpose(1,0,2))/rho
        
        
        feq = LBM.dist_eq(Nx, Ny, u, e, cs, n, rho, W)
        fneq = f - feq
        tauab = LBM.tauab(Nx, Ny, e, n, fneq)
        fneq = LBM.dist_neq(Nx, Ny, e, cs, n, W, tauab)
        fout = LBM.collision_step(feq, fneq, omega)

#***** Transmissão *****
        f = LBM.transmissao(Nx, Ny, f, fout)
        f = LBM.condicao_wall(Nx, Ny, solido, e, n, f, fout)
        
        Forca = LBM.forca(Nx, Ny, solido, e, n, fout, f)
        cl, cd = LBM.coeficientes(Nx, Ny, D_est, uini, rho_ar, Forca)
        cl_step.append(cl); cd_step.append(cd)
        
#***** Condições de Contorno *****
#        f = LBM.condicao_periodica_paredes('Inferior', Nx, fout, f)
#        f = LBM.condicao_periodica_paredes('Superior', Nx, fout, f)
        rho, u, f = LBM.zou_he('Inferior', u, rho, u_inlet, uini, f)
        rho, u, f = LBM.zou_he('Superior', u, rho, u_inlet, uini, f)
        rho, u, f = LBM.zou_he('Entrada', u, rho, u_inlet, uini, f)
        
#        f = LBM.outflow(f)
#        f = LBM.outflow_correction(rho, f)
        
        if (step % 500 == 0): print('Step -> {}'.format(step))
        
        if (step % 100 == 0):
            u_mod = modulo_velocidade(u)
            P = pressao(rho, cs)
            fg.grafico(u_mod, step, folder_vel)
            fg.grafico(P, step, folder_pres)
        
        if (step == maxiter):
            break
        step+=1
    
    fd.save_coeficientes_step(step, folder, cl_step, cd_step)
    
    cl_mean = LBM.coeficientes_medios(300, cl_step)
    cd_mean = LBM.coeficientes_medios(300, cd_step)
    
    cl_Re.append(cl_mean); cd_Re.append(cd_mean)
    
    fim = time()
    delta_t = fim - ini
    print('Simulation Time: {0:.2f} s'.format(delta_t))
    
    print('Animating...')
    fg.animation('Velocidade',folder, folder_vel)
    fg.animation('Pressao',folder, folder_pres)
    
    print('Saving Data...')
    fd.save_parametros(Nx, Ny, D_est, Cx, Cy, tau, step, delta_t, folder)


print('Saving Coefficients...')
fd.save_coeficientes(Reynolds, cl_Re, cd_Re)
print('All Done!')
'''

fim = time()
print("Finalizado em {} segundos".format(fim - ini))