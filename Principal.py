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
ini = time()

def modulo_velocidade(u):
    return np.linalg.norm(u, axis=0).transpose()

#***** Entrada de Dados *****
L = 1       # Comprimento do túnel [m]
H = 2.5     # Altura do túnel [m]
Nx = 1000    # Número de partículas em x [Lattice units]
Ny = 400    # Número de partículas em y [Lattice units]

Cx = Nx/4       # Centro do Cilindro em x [Lattice units]
Cy = Ny/2       # Centro do Cilindro em y [Lattice units]
D_est = 80    # Diâmetro do Cilindro [Lattice units]
D = 1

Reynolds = [50]    # Reynolds Numbers
cl_Re = []
cd_Re = []
cl_step = []
cd_step = []

# Propriedades do Ar
rho_ar = 1.0 #1.21
mi_ar = 1.81e-5

Uini = 1
mode = 'Constante'
#escoamento = 'Laminar'

maxiter = 70000      # Número de Iterações

#***** D2Q9 Parameters *****
n = 9                       # Número de Direções do Lattice
c = 2                       # Lattice speed
cs = 1/np.sqrt(3)           # Velocidade do Som em unidades Lattice

#***** Lattice Constants *****
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
e = np.stack((ex, ey), axis=1)
W = np.array([16/36, 4/36, 4/36, 4/36, 4/36, 1/36, 1/36, 1/36, 1/36])

#***** Construção do Cilindro e Perfil de Velocidades *****
solido = LBM.cilindro(Nx, Ny, Cx, Cy, D_est)

for Re in Reynolds:
    print('\nRe = {}'.format(Re))
    folder, folder_imagens = fd.criar_pasta(Re)
    c, tau, omega = LBM.relaxation_parameter(L, H, Nx, Ny, D_est, c, cs, Uini, Re, mode)
#    tau, omega = LBM.relaxation_parameter(D, c, cs, Uini, Re, rho_ar)
    
    u_inlet = LBM.velocidade_lattice_units(L, H, Nx, Ny, c, Uini, mode)
    uini = Uini/c
    
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
        
        fout = LBM.condicao_solido(solido, n, f, fout)

#***** Transmissão *****
        f = LBM.transmissao(Nx, Ny, f, fout)
#        f = LBM.condicao_wall(Nx, Ny, solido, e, n, f, fout)
        
        Forca = LBM.forca(Nx, Ny, solido, e, c, n, f)
        cl, cd = LBM.coeficientes(Nx, Ny, D_est, uini, rho_ar, Forca)
        cl_step.append(cl); cd_step.append(cd)
        
#***** Condições de Contorno *****
        f = LBM.condicao_periodica_paredes(Nx, fout, f)
        rho, u, f = LBM.zou_he_entrada(u, rho, u_inlet, f)
        rho, u, f = LBM.zou_he_saida(u, rho, f)
        
        if (step % 500 == 0): print('Step -> {}'.format(step))
        
        if (step % 100 == 0):
            u_mod = modulo_velocidade(u)
            fg.grafico(u_mod, step, folder_imagens)
        
        if (step == maxiter):
            break
        step+=1
    
    fd.save_coeficientes_step(step, folder, cl_step, cd_step)
    
    fim = time()
    delta_t = fim - ini
    print('Simulation Time: {0:.2f} s'.format(delta_t))
    
    print('Animating...')
    fg.animation(folder, folder_imagens)
    
#    print('Saving Data...')
#    fd.save_parametros(Nx, Ny, r, Cx, Cy, c, tau, step, delta_t, folder)

print('Saving Coefficients...')
#fd.save_coeficientes(Reynolds, cl_s, cd_s)
print('All Done!')
    
