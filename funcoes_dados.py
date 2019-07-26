#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:09:59 2019

@author: iurk
"""
def criar_pasta(Re):
    from os import listdir, mkdir
    from os.path import isdir, join
    
    onlyfolders = [f for f in listdir() if isdir(join(f))]
    
    pasta = 'Re = ' + str(Re)
    
    if pasta not in onlyfolders:
        mkdir(pasta)
        pasta_imagens = pasta + '/%s' % 'Simulacao'
        mkdir(pasta_imagens)
    else:
        pasta_imagens = pasta + '/%s' % 'Simulacao'
    return pasta, pasta_imagens

def save_parametros(Nx, Ny, r, Cx, Cy, c, tau, step, delta_t, pasta):
    from numpy import array, savetxt
    
    path = pasta + '/%s' % 'Dados.txt'
    dominio = [Ny, Ny]
    cilindro = [2*r, Cx, Cy]
    lattice = [c, tau]
    simulation = [step, delta_t]
    
    dados = __create_text(dominio, cilindro, lattice, simulation, var1='Nx', var2='Ny', var3='D',\
                        var4='Cx', var5='Cy', var6='c', var7='tau', var8='Iterações', var9='Tempo Total')
    array(dados)
    savetxt(path, dados, fmt='%s')
    
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
        