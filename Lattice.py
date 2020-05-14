#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:41:55 2020

@author: iurk
"""

class Lattice():
    
    def __init__(self, lattice):
        from numpy import array, empty, float32, int32, stack, sqrt
        
        if lattice == 'D2Q9':
            self.q = array(9, dtype=int32)
            self.cs = array(1/sqrt(3), dtype=float32)
            self.W = empty(self.q, dtype=float32)
            
            self.W[0] = 16/36
            self.W[1:5] = 4/36
            self.W[5:self.q] = 1/36
            
            self.ex = array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=int32)
            self.ey = array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=int32)
                        
    def get_attributes(self):
        return self.q, self.cs, self.W, self.ex, self.ey
    
    def get_param_dist_eq(self):
        from numpy import float32
        
        A = 1/(self.cs**2)
        B = 1/(2*self.cs**4)
        C = 1/(2*self.cs**2)
        return float32(A), float32(B), float32(C)
        

