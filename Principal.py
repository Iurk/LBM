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

Reynolds = [220]    # Reynolds Numbers
escoamento = 'Turbulento'
Uini = 1

maxiter = 1000    # Número de Iterações
tol = 1e-5

#***** Data Visualization *****
animation = True
image = True
text = True

#***** D2Q9 Parameters *****
n = 9                       # Número de Direções do Lattice
c = 25                      # Lattice speed
cs = 1/np.sqrt(3)           # Velocidade do Som em unidades Lattice

#***** Lattice Constants *****
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
e = np.stack((ex, ey), axis=1)
W = np.array([16/36, 4/36, 4/36, 4/36, 4/36, 1/36, 1/36, 1/36, 1/36])

#***** Construção do Cilindro e Perfil de Velocidades *****
solido = LBM.cilindro(Nx, Ny, Cx, Cy, r)
u_entrada = LBM.velocidade_lattice_units(L, H, Nx, Ny, c, Uini, 'Perfil', escoamento)
u_erro = np.ones((2, Nx, Ny))

for Re in Reynolds:
    folder, folder_imagens = fd.criar_pasta(Re)
    tau, omega = LBM.relaxation_parameter(L, H, Nx, Ny, r, c, cs, Uini, Re, escoamento)
    
#***** Inicialização *****
    print('Initializing')
    feq = LBM.dist_eq(Nx, Ny, u_entrada, e, cs, n, 1.0, W)
    f = feq.copy()
    
    fig, ims = fg.create_animation()
    
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
        rho, u, fout = LBM.zou_he_entrada(u, rho, u_entrada, fout)
        
        fout = LBM.condicao_solido(solido, n, f, fout)
        
#***** Transmissão *****
        f = LBM.transmissao(Nx, Ny, f, fout)
    
        if (step % 500 == 0): print('Step -> {}'.format(step))
        
        if (step % 100 == 0):
            u_mod = modulo_velocidade(u)
            fg.grafico(u_mod, step, folder_imagens)
                
        erro_u = abs(u_erro - u)
        erro_u[:, solido] = 0
        if (np.all(erro_u < tol)) or (step == maxiter):
            break
        
        u_erro = u
        step+=1
        
    fim = time()
    delta_t = fim - ini
    print('Simulation Time: {0:.2f} s'.format(delta_t))
    
    if animation:
        print('Animating...')
        fg.animation(fig, ims, folder, folder_imagens)
    
    if text:
        print('Saving Data...')
        fd.save_parametros(Nx, Ny, r, Cx, Cy, c, tau, step, delta_t, folder)
    print('All Done!')
