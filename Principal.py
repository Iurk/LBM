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
Lx = 520    # Dimensão em x [Lattice units]
Ly = 180    # Dimensão em y [Lattice units]

Cx = Lx/4   # Centro do Cilindro em x [Lattice units]
Cy = Ly/2   # Centro do Cilindro em y [Lattice units]
r = Ly/9    # Raio do Cilindro [Lattice units]

Re = 10    # Reynolds Number
Uini = 1

maxiter = 5000    # Número de Iterações
tol = 1e-5

#***** Data Visualization *****
animation = False
image = True

#***** LBM Parameters *****
n = 9                       # Número de Direções do Lattice
c = 25                   # Lattice speed
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
#u_entrada = LBM.velocidade_lattice_units(x, y, Lx, Ly, c, Uini)
u_entrada = np.fromfunction(lambda d,x,y: (1-d)*0.04*(1.0+1e-4*np.sin(y/(Ly-1)*np.pi)),(2,Lx,Ly))
u_erro = np.ones((2, Lx, Ly))

#***** Inicialização *****
print('Initializing')
feq = LBM.dist_eq(1.0, u_entrada, e, W, cs, n, Lx, Ly)
f = feq.copy()

fig, ims = fg.create_animation()

print('Running...')
step = 0
while True:
    rho = LBM.rho(f)
    u = np.dot(e.transpose(), f.transpose(1,0,2))/rho
    
    feq = LBM.dist_eq(rho, u, e, W, cs, n, Lx, Ly)
    fneq = f - feq
    tauab = LBM.tauab(fneq, e, Lx, Ly, n)
    fneq = LBM.dist_neq(tauab, e, W, cs, n, Lx, Ly)
    fout = LBM.collision_step(feq, fneq, omega)
    
#    fout = LBM.bounce_back(fout, Ly, 'Inferior')
#    fout = LBM.bounce_back(fout, Ly, 'Superior')
    fout = LBM.outflow_saida(fout)
    rho, u, fout = LBM.zou_he_entrada(u, rho, u_entrada, fout)
    
    fout = LBM.condicao_solido(f, fout, solido, n)
    
    f = LBM.transmissao(f, fout, Lx, Ly)

    if (step % 500 == 0): print('Step -> {}'.format(step))
    
    if image:
        if (step % 100 == 0):
            u_mod = modulo_velocidade(u)
            fg.grafico(u_mod, step)
    
    if animation:
        if (step % 10 == 0):
            u_mod = modulo_velocidade(u)
            ims = fg.images(u_mod, ims=ims)
            
#    if step > 2:
#        erro_u = abs(u_erro - u)
#        erro_u[:, solido] = 0
#        if (np.all(erro_u < tol)) or (step == maxiter):
#            break
    if step == maxiter:
        break
    
    u_erro = u
    step+=1
    
print('Animating...')
if animation:
    fg.save_animation(fig, ims)
print('Done!')
fim = time()
delta_t = fim - ini
print('Simulation Time: {0:.2f} s'.format(delta_t))
