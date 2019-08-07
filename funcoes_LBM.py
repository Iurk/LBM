#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:20:30 2019

@author: iurk
"""
import numpy as np
# Arquivo de funções

# Termo de Relaxação
def relaxation_parameter(L, H, Nx, Ny, r, c, cs, Uini, Re, mode, escoamento=None):
    while True:
        if mode == 'Constante':
            u_est = Uini/c
            ni_est = (u_est*(2*r))/Re
        elif mode == 'Perfil':
            U = np.zeros((2, Nx, Ny))
            U[0,:] = __perfil_velocidade(L, H, Ny, Uini, U[0], escoamento)
            u_est = U/c
            umax_est = max(u_est)
            ni_est = (umax_est*(2*r))/Re
            
        tau = ni_est/(cs**2) + 1/2
        omega = 1/tau
        
        if 0.5 <= tau <= 1.05*0.5:
            return c, tau, omega
        c *= 10

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
def velocidade_lattice_units(L, H, Nx, Ny, c, Uini, mode, escoamento=None):
    U = np.zeros((2, Nx, Ny))

    if mode == 'Perfil':
        U[0,:] = __perfil_velocidade(L, H, Ny, Uini, U[0], escoamento)
        
    elif mode == 'Constante':
        for xi in range(Nx):
            for yi in range(Ny):
                U[0,xi,yi] = Uini
    u = U/c
    return u

# Identificação das partículas que compõem o cilindro
def cilindro(Nx, Ny, Cx, Cy, r):
    solido = np.zeros((Nx, Ny), dtype=bool)
    
    for xi in range(Nx):
        for yi in range(Ny):
            if (xi - Cx)**2 + (yi - Cy)**2 <= r**2:
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
def collision_step(feq, fneq, omega):
    fout = feq +(1 - omega)*fneq
    return fout

# Transmissão
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

# Condições de Contorno
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

def condicao_wall(Nx, Ny, solido, e, n, fout):
    noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    
    for xi in range(Nx-1):
        for yi in range(Ny-1):
            for i in range(n):
                x_next = xi + e[i, 0]
                y_next = yi + e[i, 1]
                
                if solido[x_next, y_next]:
                    fout[i, xi, yi] = fout[noslip[i], xi, yi]
    return fout
    
def condicao_solido(solido, n, f, fout):
    noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    for i in range(n):
        fout[i, solido] = f[noslip[i], solido]
    return fout

def parede_cilindro(Nx, Ny, solido, e, n):
    fluid = np.zeros((Nx, Ny), dtype=bool)
    solido_l1 = np.zeros((Nx, Ny), dtype=bool)
    
    for xi in range(Nx-1):
        for yi in range(Ny-1):
            if solido[xi, yi]:
                for i in range(n):
                    x_next = xi + e[i, 0]
                    y_next = yi + e[i, 1]
                    if not solido[x_next, y_next]:
                        fluid[x_next, y_next] = True
                        solido_l1[xi, yi] = True
    return fluid, solido_l1
    
# Cálculo de Força
def forca(Nx, Ny, solido, u, e, c, n, rho, W, tau, f):
    delta = 0.5
    noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    Force = np.zeros((2))
    
    for xi in range(Nx - 1):
        for yi in range(Ny - 1):
            if solido[xi, yi]:
                Momentum = np.zeros((2))
                ubf = __ubf_definition(xi, yi, solido, u, e, n)
                for i in range(n):
                    x_next = xi + e[i, 0]
                    y_next = yi + e[i, 1]
                    
                    if not solido[x_next, y_next]:
                        chi = (2*delta - 1)/(tau - 1)
                        uf = u[:, x_next, y_next]
                        
                        fi_est = __dist_equilibrio_fic(uf, ubf, e[i], rho[x_next, y_next], W[i])
                        fi_barra = (1 - chi)*f[i, x_next, y_next] + chi*fi_est
                        
                        for a in range(2):
                            Momentum[a] += (fi_barra + f[i, x_next, y_next])*e[i, a]*c
                Force += Momentum
    return Force

def __ubf_definition(xi, yi, solido, u, e, n):
    ubf = np.zeros((2))
    
    for i in range(n):
        x_next = xi + e[i, 0]
        y_next = yi + e[i, 1]
        
        if (e[i, 0] < 0) and not solido[x_next, y_next]:
            ubf = u[:, x_next-1, y_next+1]
            break
        elif (e[i, 0] > 0) and not solido[x_next, y_next]:
            ubf = u[:, x_next, y_next+1]
            break
        elif (e[i,1] < 0) and not solido[x_next, y_next]:
            ubf = u[:, x_next, y_next-1]
            break
        elif (e[i, 1] > 0) and not solido[x_next, y_next]:
            ubf = u[:, x_next, y_next+1]
            break
    return ubf
            
            
    

def __dist_equilibrio_fic(uf, ubf, e, rho, W):
    A = np.dot(e, ubf)
    B = np.dot(e, uf)
    C = np.dot(uf, uf)
        
    feq_fic = W*rho*(1 + 3*A + (9/2)*(B**2) - (3/2)*C)
    return feq_fic

# Cálculo dos Coeficientes
def coeficientes(Nx, Ny, D, U, rho, Force):
    
    pressao_dinamica = (1/2)*rho*(U**2)
    
    cd = Force[0]/(pressao_dinamica*D)
    cl = Force[1]/(pressao_dinamica*D)
    
#    cd_avg = np.mean(cd[parede])
#    cl_avg = np.mean(cl[parede])
    return cl, cd
