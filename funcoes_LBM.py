#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:20:30 2019

@author: iurk
"""
import numpy as np
# Arquivo de funções

# Termo de Relaxação
def relaxation_parameter(L, H, Nx, Ny, r, c, cs, Uini, Re, escoamento):
    Ux = np.zeros((Nx, Ny))
    Ux = __perfil_velocidade(L, H, Ny, Uini, Ux, escoamento)
    Ux_max = max(Ux[0,:])
    
    ux_est = Ux_max/c
    ni_est = (ux_est*(r))/Re
    tau = ni_est/(cs**2) + 1/2
    omega = 1/tau
    return tau, omega

# Perfis de Velocidade
def __perfil_velocidade(L, H, Ny, Uini, Uvar, escoamento):
    mi = 1
    dP = 1
    
    if escoamento == 'Laminar':
        A = 1
    elif escoamento == 'Turbulento':
        A = 1e-4
            
    y = np.linspace(0, H, num=Ny)
    
    for yi in range(Ny):
        Uvar[:, yi] = ((H**2)/(2*mi))*(-dP/L)*((y[yi]/H)**2 - (y[yi]/H))*A + Uini
    return Uvar

# Velocidade na unidade do método e em formato Matricial 3D
def velocidade_lattice_units(L, H, Nx, Ny, c, Uini, mode, escoamento):
    U = np.zeros((2, Nx, Ny))

    if mode == 'Perfil':
        U[0,:] = __perfil_velocidade(L, H, Ny, Uini, U[0], escoamento)
        
    elif mode == 'Constante':
        for i in range(Nx):
            for j in range(Ny):
                U[0,i,j] = Uini
    u = U/c
    return u

# Identificação das partículas que compõem o cilindro
def cilindro(Nx, Ny, Cx, Cy, r):
    solido = np.zeros((Nx, Ny), dtype=bool)
    
    for x in range(Nx):
        for y in range(Ny):
            if (x - Cx)**2 + (y - Cy)**2 < r**2:
                solido[x, y] = True
    return solido

# Rho
def rho(f):
    return np.sum(f, axis=0)

# Distribuição de Equilíbrio
def dist_eq(Nx, Ny, u, e, cs, n, rho, W):
    delab = np.array([[1,0],[0,1]])
    
    A = np.zeros((n, Nx, Ny))
#    A = np.zeros((n, Nx, Ny), dtype=np.longdouble)
    for i in range(n):
        Aaux = 0
#        Aaux = np.longdouble(0)
        for a in range(2):
            Aaux += u[a]*e[i,a]
        A[i,:,:] = Aaux
    
    B = np.zeros((n, Nx, Ny))
#    B = np.zeros((n, Nx, Ny), dtype=np.longdouble)
    for i in range(n):
        Baux = 0
#        Baux = np.longdouble(0)
        for a in range(2):
            sum1 = 0
#            sum1 = np.longdouble(0)
            for b in range(2):
                sum1 += u[a]*u[b]*(e[i,a]*e[i,b] - (cs**2)*delab[a,b])
            Baux += sum1
        B[i,:,:] = Baux
        
    feq = np.zeros((n, Nx, Ny))
#    feq = np.zeros((n, Nx, Ny), dtype=np.longdouble)
    for i in range(n):
        feq[i,:,:] = W[i]*rho*(1 + (1/cs**2)*A[i] + (1/(2*cs**4))*B[i])
    return feq

# Distribuição de Não Equilíbrio
def dist_neq(Nx, Ny, e, cs, n, W, tauab):
    delab = np.array([[1,0],[0,1]])
    
    A = np.zeros((n, Nx, Ny))
#    A = np.zeros((n, Nx, Ny), dtype=np.longdouble)
    for i in range(n):
        Aaux = 0
#        Aaux = np.longdouble(0)
        for a in range(2):
            sum1 = 0
#            sum1 = np.longdouble(0)
            for b in range(2):
                sum1 += tauab[a,b]*(e[i,a]*e[i,b] - (cs**2)*delab[a,b])
            Aaux += sum1
        A[i,:,:] = Aaux
       
    fneq = np.zeros((n, Nx, Ny))
#    fneq = np.zeros((n, Nx, Ny), dtype=np.longdouble)
    for i in range(n):
        fneq[i,:,:] = W[i]*(1/(2*cs**4))*A[i]
    return fneq
    
# Momentos de Não Equilibrio
def tauab(Nx, Ny, e, n, fneq):
    tauab = np.zeros((2, 2, Nx, Ny))
#    tauab = np.zeros((2, 2, X, Y), dtype=np.longdouble)
    for a in range(2):
        for b in range(2):
            for i in range(n):
                tauab[a,b,:,:] += fneq[i]*e[i,a]*e[i,b]
    return tauab
    
def collision_step(feq, fneq, omega):
    fout = feq +(1 - omega)*fneq
    return fout

def transmissao(Nx, Ny, f, fout):
    f[0,:,:] = fout[0,:,:]
    f[1,1:Nx,:] = fout[1,0:Nx-1,:]
    f[2,:,0:Ny-1] = fout[2,:,1:Ny]
    f[3,0:Nx-1,:] = fout[3,1:Nx,:]
    f[4,:,1:Ny] = fout[4,:,0:Ny-1]
    f[5,1:Nx,0:Ny-1] = fout[5,0:Nx-1,1:Ny]
    f[6,0:Nx-1,0:Ny-1] = fout[6,1:Nx,1:Ny]
    f[7,0:Nx-1,1:Ny] = fout[7,1:Nx,0:Ny-1]
    f[8,1:Nx,1:Ny] = fout[8,0:Nx-1,0:Ny-1]
    return f   

def zou_he_entrada(u, rho, u_entrada, f):
    u[:,0,:] = u_entrada[:,0,:]
    rho[0,:] = (f[0,0,:] + f[2,0,:] + f[4,0,:] + 2*(f[3,0,:] + f[6,0,:] + f[7,0,:]))/(1 - u[0,0,:])
    
    f[1,0,:] = f[3,0,:] + (2/3)*rho[0,:]*u[0,0,:]
    f[5,0,:] = f[7,0,:] - (1/2)*(f[2,0,:] - f[4,0,:]) + (1/6)*rho[0,:]*u[0,0,:]
    f[8,0,:] = f[6,0,:] + (1/2)*(f[2,0,:] - f[4,0,:]) + (1/6)*rho[0,:]*u[0,0,:]
    return rho, u, f

def outflow_saida(f):
    unknow = [3,6,7]
    f[unknow,-1,:] = f[unknow,-2,:]
    return f

def bounce_back(f, parede):
    if parede == 'Superior':
        f[4,:,-1] = f[2,:,-1]
        f[7,:,-1] = f[5,:,-1]
        f[8,:,-1] = f[6,:,-1]
        
    elif parede == 'Inferior':
        f[2,:,0] = f[4,:,0]
        f[5,:,0] = f[7,:,0]
        f[6,:,0] = f[8,:,0]
    return f

def condicao_solido(solido, n, f, fout):
    noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    for i in range(n):
        fout[i, solido] = f[noslip[i], solido]
    return fout
    