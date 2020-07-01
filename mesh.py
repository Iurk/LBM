#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:49:54 2020

@author: iurk
"""
import yaml
import numpy as np

fileyaml = "./bin/dados.yml"

datafile = open(fileyaml)
data = yaml.load(datafile, Loader=yaml.FullLoader)

Nx = data['domain']['Nx']
Ny = data['domain']['Ny']

Cx = data['cylinder']['Cx']
Cy = data['cylinder']['Cy']
D = data['cylinder']['D']

Cx = eval(Cx)
Cy = eval(Cy)
           
solid = np.zeros(Nx*Ny, dtype=bool)
for y in range(Ny):
    for x in range(Nx):
        if (x - Cx)**2 + (y - Cy)**2 <= (D/2)**2:
            solid[Nx*y + x] = True
        
file = open('./bin/mesh.bin', 'wb')
file.write(bytearray(solid))
file.close()
