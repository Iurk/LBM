#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:19:27 2019

@author: iurk
"""
import numpy as np
from time import time
import funcoes_LBM as LBM
import funcoes_graficos as fg
ini = time()

def modulo_velocidade(u):
    return np.linalg.norm(u, axis=0).transpose()

#***** Entrada de Dados *****
Lx = 700    # Dimensão em x [Lattice units]
Ly = 200    # Dimensão em y [Lattice units]

Cx = Lx/4   # Centro do Cilindro em x [Lattice units]
Cy = Ly/2   # Centro do Cilindro em y [Lattice units]
r = Ly/10    # Raio do Cilindro [Lattice units]

Re = 220    # Reynolds Number
Uini = 5

maxiter = 3000    # Número de Iterações

#***** LBM Parameters *****
n = 9                       # Número de Direções do Lattice
c = 2e2                     # Lattice speed
cs = 1/np.sqrt(3)           # Velocidade do Som em unidades Lattice
tau, omega = LBM.relaxation_parameter(Re, r, Lx, Ly, c, Uini)

#***** Lattice Constants *****
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
e = np.stack((ex, ey), axis=1)
W = np.array([16/36, 4/36, 4/36, 4/36, 4/36, 1/36, 1/36, 1/36, 1/36])

# Distribuição das Partículas
x = np.arange(0, Lx)
y = np.arange(0, Ly)

#***** Construção do Cilindro e Perfil de Velocidades *****
solido = np.array([LBM.cilindro(i, y, Cx, Cy, r) for i in x])
u_entrada = LBM.velocidade_lattice_units(x, y, Lx, Ly, c, Uini)

#***** Inicialização *****
print('Initializing')
feq = LBM.dist_eq(1.0, u_entrada, e, W, cs, n, Lx, Ly)
f = feq.copy()

fig, ims = fg.create_animation()

print('Running...')
#***** Main Loop *****
for step in range(maxiter):
    rho = LBM.sum_directions(f)
    u = np.dot(e.transpose(), f.transpose(1,0,2))/rho
    
    feq = LBM.dist_eq(rho, u, e, W, cs, n, Lx, Ly)
    fneq = f - feq
    tauab = LBM.tauab(fneq, e, Lx, Ly, n)
    fneq = LBM.dist_neq(tauab, e, W, cs, n, Lx, Ly)
    fout = LBM.collision_step(feq, fneq, tau)
    
    rho, u, fout = LBM.zou_he_entrada(u, rho, u_entrada, fout)
#    rho, u, fout = LBM.zou_he_saida(u, rho, u_entrada, fout)
    fout = LBM.outflow_saida(fout)
    fout = LBM.bounce_back(fout, Ly, 'Superior')
    fout = LBM.bounce_back(fout, Ly, 'Inferior')
    
    fout = LBM.condicao_solido(f, fout, solido, n)
    
    f = LBM.transmissao(f, fout, Lx, Ly)
    
#    f = LBM.new_transmissao(f, fout, n)
    
#    for i in range(n):
#        f[i,:,:] = np.roll(np.roll(fout[i,:,:], e[i,0], axis=0), e[i,1], axis=1)
    
    if (step % 500 == 0): print('Step -> {}'.format(step))
    
    if (step % 100 == 0):
        u_mod = modulo_velocidade(u)
        fg.grafico(u_mod, step)
        
    if (step % 10 == 0):
        u_mod = modulo_velocidade(u)
        ims = fg.images(u_mod, ims=ims)

print('Animating...')
fg.save_animation(fig, ims)
print('Done!')
fim = time()
delta_t = fim - ini
print('Simulation Time: {0:.2f} s'.format(delta_t))
