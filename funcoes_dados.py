#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:09:59 2019

@author: iurk
"""
def criar_pasta(Re):
    from shutil import rmtree
    from os import listdir, mkdir, getcwd
    from os.path import isdir, join
    
    onlyfolders = [f for f in listdir(getcwd()) if isdir(join(f))]
    
    pasta = 'Re = ' + str(Re)
    pasta_imagens = pasta + '/%s' % 'Simulacao'
    pasta_vel = pasta_imagens + '/%s' % 'Velocidade'
    pasta_pres = pasta_imagens + '/%s' % 'Pressao'
    
    if pasta not in onlyfolders:
        mkdir(pasta)
        mkdir(pasta_imagens)
        mkdir(pasta_vel)
        mkdir(pasta_pres)
    else:
        rmtree(pasta_imagens)
        mkdir(pasta_imagens)
        mkdir(pasta_vel)
        mkdir(pasta_pres)
        
    return pasta, pasta_vel, pasta_pres

def save_parametros(Nx, Ny, D, Cx, Cy, tau, step, delta_t, pasta):
    from numpy import array, savetxt
    
    path = pasta + '/%s' % 'Dados.txt'
    dominio = [Ny, Ny]
    cilindro = [D, Cx, Cy]
    lattice = [tau]
    simulation = [step, delta_t]
    
    dados = __create_text(dominio, cilindro, lattice, simulation, var1='Nx', var2='Ny', var3='D',\
                        var4='Cx', var5='Cy', var6='tau', var7='Iterações', var8='Tempo Total')
    array(dados)
    savetxt(path, dados, fmt='%s')
    
def save_coeficientes_step(steps, pasta, cl, cd):
    from numpy import array, savetxt
    
    file = 'Coeficientes_step.txt'
    path = pasta + '/%s' % file
    
    s = []
    title = 'Step \t cl \t cd'
    
    s.append(title)
    for i in range(steps):
        saux = '{:d} \t {:.3f} \t {:.3f}'.format(i, cl[i], cd[i])
        s.append(saux)
    
    array(s)
    savetxt(path, s, fmt='%s')
    
def save_coeficientes(Re, cl, cd):
    from numpy import array, savetxt
    
    file = 'Coeficientes.txt'
    
    s = []
    title = 'Re \t  cl \t  cd'
    
    s.append(title)
    for i in range(len(Re)):
        saux = '{:d} \t {:.3f} \t {:.3f}'.format(Re[i], cl[i], cd[i])
        s.append(saux)
        
    array(s)
    savetxt(file, s, fmt='%s')    
    
def __create_text(dominio, cilindro, lattice, simulation, **kwargs):
    values = list(kwargs.values())
    s = []
    
    title1 = 'Domínio'
    title2 = 'Cilíndro'
    title3 = 'Lattice Parameters'
    title4 = 'Geral'
    
    cont = 0
    s.append(title1)
    for i in dominio:
        saux = '{} = {}'.format(values[cont], i)
        s.append(saux)
        cont += 1
    
    s.append('')
    s.append(title2)
    for i in cilindro:
        saux = '{} = {}'.format(values[cont], i)
        s.append(saux)
        cont += 1
        
    s.append('')    
    s.append(title3)
    for i in lattice:
        if values[cont] == 'tau':
            saux = str(values[cont]) + ' = {0:.2f}'.format(i)
        else:
            saux = '{} = {}'.format(values[cont], i)
        s.append(saux)
        cont += 1
        
    s.append('')    
    s.append(title4)
    for i in simulation:
        if values[cont] == 'Tempo Total':
            saux = str(values[cont]) + ' = {0:.2f}'.format(i)
        else:
            saux = '{} = {}'.format(values[cont], i)
        s.append(saux)
        cont += 1
    return s
        