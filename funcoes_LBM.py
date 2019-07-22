#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:20:30 2019

@author: iurk
"""
import numpy as np
# Arquivo de funções

# Termo de Relaxação
def relaxation_parameter(Re, r, X, Y, c, Uini):
    
    cs = 1/np.sqrt(3)
    Ux = [perfil_velocidade(y, X, Y, Uini) for y in range(Y)]
    Ux_max = max(Ux)
    
    ux_est = Ux_max/c
    ni_est = (ux_est*(2*r))/Re
    tau = ni_est/(cs**2) + 1/2
    omega = 1/tau
    return tau, omega

# Perfil de Velocidade
def perfil_velocidade(y, X, Y, Uini):
    mi = 1
    dP = 1
    
    U = (((Y-1)**2)/(2*mi))*(-dP/(X))*((y/(Y-1))**2 - (y/(Y-1))) + Uini
    return U

# Velocidade na unidade do método e em formato Matricial 3D
def velocidade_lattice_units(x, y, X, Y, c, Uini, mode='Default'):
    U = np.zeros((2, X, Y))
    Ux = []

    if mode == 'Default':
        for i in x:
            Ux_aux = [perfil_velocidade(j, X, Y, Uini) for j in y]
            Ux.append(Ux_aux)
        
        Ux = np.array(Ux)
        Uy = np.zeros((X, Y))
        
        U[0,:] = Ux
        U[1,:] = Uy
        
    else:
        for i in range(X):
            for j in range(Y):
                U[0,i,j] = Uini
    
    u = U/c
    return u

# Identificação das partículas que compõem o cilindro
def cilindro(x, y, Cx, Cy, r):
    return (x - Cx)**2 + (y - Cy)**2 < r**2

# Rho e u
def sum_directions(f):
    return np.sum(f, axis=0)

def u(e, f, rho, n, X, Y):
    u = np.zeros((2, X, Y))
    
    for i in range(X):
        for j in range(Y):
            for k in range(n):
                u[0,i,j] += e[k,0]*f[k,i,j]
                u[1,i,j] += e[k,1]*f[k,i,j]

    u = u/rho
    return u
        
# Distribuição de Equilíbrio
def dist_eq(rho, u, e, W, cs, n, X, Y):
    delab = np.array([[1,0],[0,1]])
    
    A = np.zeros((n, X, Y))
    for i in range(n):
        Aaux = 0
        for a in range(2):
            Aaux += u[a]*e[i,a]
        A[i,:,:] = Aaux
    
    B = np.zeros((n, X, Y))
    for i in range(n):
        Baux = 0
        for a in range(2):
            sum1 = 0
            for b in range(2):
                sum1 += u[a]*u[b]*(e[i,a]*e[i,b] - (cs**2)*delab[a,b])
            Baux += sum1
        B[i,:,:] = Baux
        
    feq = np.zeros((n, X, Y))
    for i in range(n):
        feq[i,:,:] = W[i]*rho*(1 + (1/cs**2)*A[i] + (1/(2*cs**2))*B[i])
    return feq

# Distribuição de Não Equilíbrio
def dist_neq(tauab, e, W, cs, n, X, Y):
    delab = np.array([[1,0],[0,1]])
    
    A = np.zeros((n, X, Y))
    for i in range(n):
        Aaux = 0
        for a in range(2):
            sum1 = 0
            for b in range(2):
                sum1 += tauab[a,b]*(e[i,a]*e[i,b] - (cs**2)*delab[a,b])
            Aaux += sum1
        A[i,:,:] = Aaux
        
    fneq = np.zeros((n, X, Y))
    for i in range(n):
        fneq[i,:,:] = W[i]*(1/(2*cs**4))*A[i]
    return fneq
    
# Momentos de Não Equilibrio
def tauab(fneq, e, X, Y, n):
    tauab = np.zeros((2, 2, X, Y))
    for a in range(2):
        for b in range(2):
            for i in range(n):
                tauab[a,b,:,:] += fneq[i]*e[i,a]*e[i,b]
    return tauab
    
def collision_step(feq, fneq, omega):
    fout = feq +(1 - omega)*fneq
    return fout

def transmissao(f, fout, X, Y):
    f[0,:,:] = fout[0,:,:]
    f[1,1:X,:] = fout[1,0:X-1,:]
    f[2,:,0:Y-1] = fout[2,:,1:Y]
    f[3,0:X-1,:] = fout[3,1:X,:]
    f[4,:,1:Y] = fout[4,:,0:Y-1]
    f[5,1:X,0:Y-1] = fout[5,0:X-1,1:Y]
    f[6,0:X-1,0:Y-1] = fout[6,1:X,1:Y]
    f[7,0:X-1,1:Y] = fout[7,1:X,0:Y-1]
    f[8,1:X,1:Y] = fout[8,0:X-1,0:Y-1]
    return f

def new_transmissao(f, fout, n):
    ex_roll = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    ey_roll = np.array([0, 0, -1, 0, 1, -1, -1, 1, 1])
    e_roll = np.stack((ex_roll, ey_roll), axis=1)
    
    for i in range(n):
        f[i,:,:] = np.roll(np.roll(fout[i,:,:], e_roll[i,0], axis=1), e_roll[i,1], axis=0)
    return f
    

def zou_he_entrada(u, rho, u_entrada, f):
    u[:,0,:] = u_entrada[:,0,:]
    rho[0,:] = (f[0,0,:] + f[2,0,:] + f[4,0,:] + 2*(f[3,0,:] + f[6,0,:] + f[7,0,:]))/(1 - u[0,0,:])
    
    f[1,0,:] = f[3,0,:] + (2/3)*rho[0,:]*u[0,0,:]
    f[5,0,:] = f[7,0,:] - (1/2)*(f[2,0,:] - f[4,0,:]) + (1/6)*rho[0,:]*u[0,0,:]
    f[8,0,:] = f[6,0,:] + (1/2)*(f[2,0,:] - f[4,0,:]) + (1/6)*rho[0,:]*u[0,0,:]
    return rho, u, f

def zou_he_saida(u, rho, u_entrada, f):
    u[:,:,-1] = u_entrada[:,:,0]
    rho[:,-1] = (f[0,:,-1] + f[2,:,-1] + f[4,:,-1] + 2*(f[1,:,-1] + f[5,:,-1] + f[8,:,-1]))/(1 + u[0,:,-1])
    
    f[3,:,-1] = f[1,:,-1] - (2/3)*rho[:,-1]*u[0,:,-1]
    f[6,:,-1] = f[8,:,-1] - (1/2)*(f[2,:,-1] - f[4,:,-1]) - (1/6)*rho[:,-1]*u[0,:,-1]
    f[7,:,-1] = f[5,:,-1] + (1/2)*(f[2,:,-1] - f[4,:,-1]) - (1/6)*rho[:,-1]*u[0,:,-1]
    return rho, u, f
    
def bounce_back(f, Y, parede):
    if parede == 'Superior':
        f[4,:,Y-1] = f[2,:,Y-1]
        f[7,:,Y-1] = f[5,:,Y-1]
        f[8,:,Y-1] = f[6,:,Y-1]
        
    elif parede == 'Inferior':
        f[2,:,0] = f[4,:,0]
        f[5,:,0] = f[7,:,0]
        f[6,:,0] = f[8,:,0]
    return f

def outflow_saida(f):
    unknow = [3,6,7]
    f[unknow,-1,:] = f[unknow,-2,:]
    return f

def condicao_solido(f, fout, solido, n):
    noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    for i in range(n):
        fout[i, solido] = f[noslip[i], solido]
    return fout
    