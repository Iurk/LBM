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
Nx = 520    # Número de partículas em x [Lattice units]
Ny = 180    # Número de partículas em y [Lattice units]

Cx = Nx/4   # Centro do Cilindro em x [Lattice units]
Cy = Ny/2   # Centro do Cilindro em y [Lattice units]
r = Ny/9    # Raio do Cilindro [Lattice units]
D = 2*r

Reynolds = [50]    # Reynolds Numbers
cl_s = []
cd_s = []
cl_step = []
cd_step = []

Uini = 1.0
mode = 'Constante'
#escoamento = 'Laminar'

maxiter = 3000      # Número de Iterações
tol = 1e-5          # Tolerância para Convergência

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
solid = LBM.cilindro(Nx, Ny, Cx, Cy, r)
wall = LBM.parede_cilindro(Nx, Ny, solid, e, n)

for Re in Reynolds:
    print('\nRe = {}'.format(Re))
    folder, folder_imagens = fd.criar_pasta(Re)
    c, tau, omega = LBM.relaxation_parameter(L, H, Nx, Ny, r, c, cs, Uini, Re, mode)
    
    u_inlet = LBM.velocidade_lattice_units(L, H, Nx, Ny, c, Uini, mode)
    u_erro = np.ones((2, Nx, Ny))
    
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
        
#***** Condições de Contorno *****
        fout = LBM.bounce_back(fout, 'Inferior')
        fout = LBM.bounce_back(fout, 'Superior')
        fout = LBM.outflow_saida(fout)
        rho, u, fout = LBM.zou_he_entrada(u, rho, u_inlet, fout)
        
        fout = LBM.condicao_solido(solid, n, f, fout)

#***** Transmissão *****
        f = LBM.transmissao(Nx, Ny, f, fout)

        Forca = LBM.forca(Nx, Ny, solid, u, e, c, n, rho, W, tau, f)
        cl, cd = LBM.coeficientes(Nx, Ny, D, Uini, 1.0, Forca)
        cl_step.append(cl); cd_step.append(cd)
        
        if (step % 500 == 0): print('Step -> {}'.format(step))
        
        if (step % 100 == 0):
            u_mod = modulo_velocidade(u)
            fg.grafico(u_mod, step, folder_imagens)
                
#        erro_u = abs(u_erro - u)
#        erro_u[:, solid] = 0
#        if (np.all(erro_u < tol)) or (step == maxiter):
#            break
        
        if (step == maxiter):
            break
        u_erro = u
        step+=1
    
    fd.save_coeficientes_step(step, folder, cl_step, cd_step)
    
#    Forca = LBM.forca(Nx, Ny, wall, e, n, fout, f)
#    Forca2 = LBM.forca_2(Nx, Ny, wall, solid, )
#    cd, cl = LBM.coeficientes(Nx, Ny, D, wall, u_inlet, rho, Forca)
#    
#    cd_s.append(cd); cl_s.append(cl)
    
    fim = time()
    delta_t = fim - ini
    print('Simulation Time: {0:.2f} s'.format(delta_t))
    
    print('Animating...')
    fg.animation(folder, folder_imagens)
    
    print('Saving Data...')
    fd.save_parametros(Nx, Ny, r, Cx, Cy, c, tau, step, delta_t, folder)

print('Saving Coefficients...')
#fd.save_coeficientes(Reynolds, cl_s, cd_s)
print('All Done!')
    
