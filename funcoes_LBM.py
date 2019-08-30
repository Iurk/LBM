#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:20:30 2019

@author: iurk
"""
import numpy as np
# Arquivo de funções

# Termo de Relaxação
def relaxation_parameter(L, H, Nx, Ny, D, cs, uini, Re, mode, escoamento=None):
    ni_est = (uini*D)/Re
    tau = ni_est/(cs**2) + 1/2
    omega = 1/tau
    return tau, omega

# Perfis de Velocidade
def __perfil_velocidade(Nx, Ny, uini, uvar, escoamento):
    mi = 1
    dP = 1
    
    if escoamento == 'Laminar':
        A = 1
    elif escoamento == 'Turbulento':
        A = 1e-6
    
    for yi in range(Ny):
        uvar[:, yi] = ((Ny**2)/(2*mi))*(-dP/Nx)*((yi/(Ny-1))**2 - (yi/(Ny-1)))*A + uini
    return uvar

# Velocidade na unidade do método e em formato Matricial 3D
def velocidade_lattice_units(Nx, Ny, uini, mode, escoamento=None):
    u = np.zeros((2, Nx, Ny))

    if mode == 'Perfil':
        u[0,:] = __perfil_velocidade(Nx, Ny, uini, u[0], escoamento)
        
    elif mode == 'Constante':
        for xi in range(Nx):
            for yi in range(Ny):
                u[0,xi,yi] = uini
    return u

# Identificação das partículas que compõem o cilindro
def cilindro(Nx, Ny, Cx, Cy, D):
    solido = np.zeros((Nx, Ny), dtype=bool)
    
    for xi in range(Nx):
        for yi in range(Ny):
            if (xi - Cx)**2 + (yi - Cy)**2 <= (D/2)**2:
                solido[xi, yi] = True
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
    
# Etapa de Colisão
def collision_step(feq, fneq, omega):  #feq, fneq, omega
    fout = feq +(1 - omega)*fneq
#    fout = omega*feq + (1 - omega)*f
    return fout

# Transmissão
def transmissao(Nx, Ny, f, fout):
    f[0,:,:] = fout[0,:,:]
    f[1,1:Nx,:] = fout[1,0:Nx-1,:]
    f[2,:,1:Ny] = fout[2,:,0:Ny-1]
    f[3,0:Nx-1,:] = fout[3,1:Nx,:]
    f[4,:,0:Ny-1] = fout[4,:,1:Ny]
    f[5,1:Nx,1:Ny] = fout[5,0:Nx-1,0:Ny-1]
    f[6,0:Nx-1,1:Ny] = fout[6,1:Nx,0:Ny-1]
    f[7,0:Nx-1,0:Ny-1] = fout[7,1:Nx,1:Ny]
    f[8,1:Nx,0:Ny-1] = fout[8,0:Nx-1,1:Ny]
    return f

# Condições de Contorno
def zou_he_entrada(u, rho, u_entrada, f):
    u[:,0,:] = u_entrada[:,0,:]
    rho[0,:] = (f[0,0,:] + f[2,0,:] + f[4,0,:] + 2*(f[3,0,:] + f[6,0,:] + f[7,0,:]))/(1 - u[0,0,:])
    
    f[1,0,:] = f[3,0,:] + (2/3)*rho[0,:]*u[0,0,:]
    f[5,0,:] = f[7,0,:] - (1/2)*(f[2,0,:] - f[4,0,:]) + (1/6)*rho[0,:]*u[0,0,:]
    f[8,0,:] = f[6,0,:] + (1/2)*(f[2,0,:] - f[4,0,:]) + (1/6)*rho[0,:]*u[0,0,:]
    return rho, u, f

def zou_he_saida(u, rho, f):
    rho[-1,:] = 1.0
    u[0,-1,:] = (f[0,-1,:] + f[2,-1,:] + f[4,-1,:] + 2*(f[1,-1,:] + f[5,-1,:] + f[8,-1,:]))/rho[-1,:] - 1
    u[1,-1,:] = 0
    
    f[3,-1,:] = f[1,-1,:] - (2/3)*rho[-1,:]*u[0,-1,:]
    f[6,-1,:] = f[8,-1,:] - (1/2)*(f[2,-1,:] - f[4,-1,:]) - (1/6)*rho[-1,:]*u[0,-1,:]
    f[7,-1,:] = f[5,-1,:] + (1/2)*(f[2,-1,:] - f[4,-1,:]) - (1/6)*rho[-1,:]*u[0,-1,:]
    return rho, u, f

def extrapolacao_saida(f):
    unknow = [3,6,7]
    f[unknow,-1,:] = 2*f[unknow,-2,:] - f[unknow,-3,:]
    return f
    
def outflow(f):
    unknow = [3,6,7]
    f[unknow,-1,:] = f[unknow,-2,:]
    return f

def outflow_correction(rho, f):
    unknow = [3,6,7]
    
    fator = 1/rho[-1,:]
    f[unknow,-1,:] = f[unknow,-2,:]*fator
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

def condicao_periodica_paredes(Nx, fout, f):
    # Inferior
    f[2,:,0] = fout[2,:,-1]
    f[5,1:Nx,0] = fout[5,0:Nx-1,-1]
    f[6,0:Nx-1,0] = fout[6,1:Nx,-1]
    
    #Superior
    f[4,:,-1] = fout[4,:,0]
    f[7,0:Nx-1,-1] = fout[7,1:Nx,0]
    f[8,1:Nx,-1] = fout[8,0:Nx-1,0]
    return f    

def condicao_wall(Nx, Ny, solido, e, n, f, fout):
    noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    
    fluid_wall = __parede_cilindro(Nx, Ny, solido, e, n)
    for xi in range(Nx-1):
        for yi in range(Ny-1):
            if fluid_wall[xi, yi]:
                for i in range(n):
                    x_next = xi + e[i, 0]
                    y_next = yi + e[i, 1]
                    
                    if solido[x_next, y_next]:
                        f[noslip[i], xi, yi] = fout[i, xi, yi]
    return f
    
def condicao_solido(solido, n, f, fout):
    noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    for i in range(n):
        fout[i, solido] = f[noslip[i], solido]
    return fout

def __parede_cilindro(Nx, Ny, solido, e, n):
    fluid = np.zeros((Nx, Ny), dtype=bool)
    
    for xi in range(Nx-1):
        for yi in range(Ny-1):
            if solido[xi, yi]:
                for i in range(n):
                    x_next = xi + e[i, 0]
                    y_next = yi + e[i, 1]
                    if not solido[x_next, y_next]:
                        fluid[x_next, y_next] = True
    return fluid
    
# Cálculo de Força
def forca(Nx, Ny, solido, e, n, f_before, f_after):
    noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    Force = np.zeros((2))
    
    fluid_wall = __parede_cilindro(Nx, Ny, solido, e, n)
    for xi in range(Nx - 1):
        for yi in range(Ny - 1):
            if fluid_wall[xi, yi]:
                Momentum = np.zeros((2))
                for i in range(n):
                    x_next = xi + e[i,0]
                    y_next = yi + e[i,1]
                    
                    if solido[x_next, y_next]:
                        for a in range(2):
                            Momentum[a] += (f_after[noslip[i],xi,yi] + f_before[i,xi,yi])*e[i,a]
                Force += Momentum
    return Force

# Cálculo dos Coeficientes
def coeficientes(Nx, Ny, D_est, u, rho, Force):
    
    Area = 1*D_est
    pressao_dinamica = (1/2)*rho*(u**2)
    
    cd = Force[0]/(pressao_dinamica*Area)
    cl = Force[1]/(pressao_dinamica*Area)
    return cl, cd

def coeficientes_medios(num, coeff):
    ini = len(coeff) - num - 1
    fim = len(coeff) - 1
    
    vetor = coeff[ini:fim]
    valor_medio = np.mean(vetor)
    return valor_medio
    
