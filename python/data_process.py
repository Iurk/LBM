#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:27:54 2020

@author: iurk
"""
import yaml
import numpy as np
from os import walk
import utilidades as util
import funcoes_graficos as fg

main = "./bin"
fileyaml = "./bin/dados.yml"
velocity = "Velocity"


datafile = open(fileyaml)
data = yaml.load(datafile, Loader=yaml.FullLoader)
datafile.close()

Nx = data['domain']['Nx']
Ny = data['domain']['Ny']

Steps = data['simulation']['NSTEPS']
Saves = data['simulation']['NSAVE']
digitos = len(str(Steps))

idx_files = ["%0{}d".format(digitos) % i for i in range(0, Steps+Saves, Saves)]

variables = ["rho", "ux", "uy"]
results = "./bin/Results/"

rho = np.empty((len(idx_files), Ny, Nx))
ux = np.empty_like(rho)
uy = np.empty_like(rho)
dic = {"rho": rho, "ux":ux, "uy":uy}

for var in variables:
    path = results + var
    i = 0
    print('Getting ' + var + ' data...')
    for root, dirs, files in walk(path):
        for file in files:
            path = root + "/%s" % file
            dic[var][i] = np.fromfile(path).reshape(Ny, Nx)
            i += 1
    
u_mod = np.sqrt(ux**2 + uy**2)

pasta_img = util.criar_pasta('Images', folder=velocity, main_root=main)
pasta_stream = util.criar_pasta('Stream', folder=velocity, main_root=main)

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
