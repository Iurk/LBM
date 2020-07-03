#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:27:54 2020

@author: iurk
"""
import yaml
import numpy as np
import funcoes_graficos as fg

fileyaml = "./bin/dados.yml"
pasta = "Velocity"

datafile = open(fileyaml)
data = yaml.load(datafile, Loader=yaml.FullLoader)

Nx = data['domain']['Nx']
Ny = data['domain']['Ny']

Steps = data['simulation']['NSTEPS']
Saves = data['simulation']['NSAVE']
digitos = len(str(Steps))

idx_files = ["%0{}d".format(digitos) % i for i in range(0, Steps+Saves, Saves)]

files = ["rho", "ux", "uy"]
folder = "./Results/"

rho = np.empty((len(idx_files), Ny, Nx))
ux = np.empty_like(rho)
uy = np.empty_like(rho)
dic = {"rho": rho, "ux":ux, "uy":uy}

for file in files:
    i = 0
    print('Getting ' + file + ' data...')
    for step in idx_files:
        full_path = folder + file + step + '.bin'
        dic[file][i] = np.fromfile(full_path).reshape(Ny, Nx)
        i += 1
        
u_mod = np.sqrt(ux**2 + uy**2)

pasta_img = fg.criar_pasta('Images', main_root='Velocity')
pasta_stream = fg.criar_pasta('Stream', main_root='Velocity')

print('Generating images...')
for i in range(len(idx_files)):
    fg.grafico(u_mod[i], idx_files[i], pasta_img)
    
print('Generating stream plots...')
x = np.arange(1, Nx+1, 1)
y = np.arange(1, Ny+1, 1)
for i in range(len(idx_files)):
    fg.stream(x, y, ux[i], uy[i], u_mod[i], idx_files[i], pasta_stream)

print('Animating...')
fg.animation('Velocidade', './', pasta_img)
fg.animation('Stream', './', pasta_stream)
print('Done!')
