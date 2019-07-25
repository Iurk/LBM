#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:09:59 2019

@author: iurk
"""

def __create_text(dominio, cilindro, lattice, **kwargs):
    values = list(kwargs.values())
    s = []
    
    title1 = 'Domínio'
    title2 = 'Cilíndro'
    title3 = 'Lattice Parameters'
    
    cont = 0
    s.append(title1)
    for i in dominio:
        saux = '{} = {}'.format(values[cont], i)
        s.append(saux)
        cont += 1
        
    s.append(title2)
    for i in cilindro:
        saux = '{} = {}'.format(values[cont], i)
        s.append(saux)
        cont += 1
        
    s.append(title3)
    for i in cilindro:
        if values[cont] == 'tau':
            saux = '{} = {0:.2f}'.format(values[cont], i)
        else:
            saux = '{} = {}'.format(values[cont], i)
        s.append(saux)
        cont += 1
    return s

def save_parametros(Lx, Ly, r, Cx, Cy, c, tau):
    from numpy import array, savetext
    
    dominio = [Lx, Ly]
    cilindro = [2*r, Cx, Cy]
    lattice = [c, tau]
    
    dados = __create_text(dominio, cilindro, lattice, var1='Lx', var2='Ly', var3='D',\
                        var4='Cx', var5='Cy', var6='c', var7='tau')
    array(dados)
    savetext('Dados.txt', dados, fmt='%s')    
        