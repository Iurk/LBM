#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:19:27 2019

@author: iurk
"""
import numpy as np
from time import time
import funcoes_LBM as LBM
import funcoes_dados as fd
import funcoes_graficos as fg

def modulo_velocidade(u):
    return np.linalg.norm(u, axis=0).transpose()

def pressao(rho, cs):
    pressao = rho*cs**2
    return pressao.transpose()

#***** Entrada de Dados *****
L = 1       # Comprimento do túnel [m]
H = 2.5     # Altura do túnel [m]
Nx = 1024    # Número de partículas em x [Lattice units]
Ny = 512    # Número de partículas em y [Lattice units]

Cx = Nx/4       # Centro do Cilindro em x [Lattice units]
Cy = Ny/2       # Centro do Cilindro em y [Lattice units]
D_est = 100    # Diâmetro do Cilindro [Lattice units]

Reynolds = [15]    # Reynolds Numbers
cl_Re = []
cd_Re = []

# Propriedades do Ar
rho_ar = 1.0 #1.21
mi_ar = 1.81e-5

uini = 0.04
mode = 'Constante'
#escoamento = 'Turbulento'

maxiter = 5000      # Número de Iterações

#***** D2Q9 Parameters *****
n = 9                       # Número de Direções do Lattice
cs = 1/np.sqrt(3)           # Velocidade do Som em unidades Lattice

#***** Lattice Constants *****
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
e = np.stack((ex, ey), axis=1)
W = np.array([16/36, 4/36, 4/36, 4/36, 4/36, 1/36, 1/36, 1/36, 1/36])

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
        u = np.dot(e.transpose(), f.transpose(1,0,2))/rho
        
        feq = LBM.dist_eq(Nx, Ny, u, e, cs, n, rho, W)
        fneq = f - feq
        tauab = LBM.tauab(Nx, Ny, e, n, fneq)
        fneq = LBM.dist_neq(Nx, Ny, e, cs, n, W, tauab)
        fout = LBM.collision_step(feq, fneq, omega)
        
        fneq_old = fneq
        
        if np.all((fneq - fneq_old) > 1e-18):
            print(step)

#***** Transmissão *****
        f = LBM.transmissao(Nx, Ny, f, fout)
        # f = LBM.condicao_wall(Nx, Ny, solido, e, n, f, fout)
        
        Forca = LBM.forca(Nx, Ny, solido, e, n, fout, f)
        cl, cd = LBM.coeficientes(Nx, Ny, D_est, uini, rho_ar, Forca)
        cl_step.append(cl); cd_step.append(cd)
        
#***** Condições de Contorno *****
#        f = LBM.condicao_periodica_paredes('Inferior', Nx, fout, f)
#        f = LBM.condicao_periodica_paredes('Superior', Nx, fout, f)
        rho, u, f = LBM.zou_he('Inferior', u, rho, u_inlet, uini, f)
        rho, u, f = LBM.zou_he('Superior', u, rho, u_inlet, uini, f)
        rho, u, f = LBM.zou_he('Entrada', u, rho, u_inlet, uini, f)
        rho, u, f = LBM.zou_he('Saída', u, rho, u_inlet, uini, f)
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
