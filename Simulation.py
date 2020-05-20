#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:54:48 2020

@author: iurk
"""
from numpy import float32, ones, zeros
class Simulation():
        
    def __init__(self, Nx, Ny, Cx, Cy, D, uini):
        self.nx = Nx
        self.ny = Ny
        self.cx = Cx
        self.cy = Cy
        self.d = D
        self.uini = uini
          
    def initialize(self):
        self.u = self.__init_u()
        self.rho = self.__init_rho()
        return self.rho, self.u
    
    def solid_create(self):
        self.solid = zeros((self.nx, self.ny), dtype=bool)
        for y in range(self.ny):
            for x in range(self.nx):
                if (x - self.cx)**2 + (y - self.cy)**2 <= (self.d/2)**2:
                    self.solid[x, y] = True
                    
    def get_solid(self):
        return self.solid
    
    def __init_u(self):
        u = zeros((2, self.nx, self.ny), dtype=float32)
        u[0, :, :] = self.uini
        return u
    
    def __init_rho(self):
        rho = ones((self.nx, self.ny), dtype=float32)
        return rho
    
    def relaxation_term(self, cs, Re):
        ni_est = (self.uini*self.d)/Re
        self.tau = ni_est/(cs**2) + 1/2
        self.omega = float32(1/self.tau)
        
    
            