#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:46:28 2020

@author: iurk
"""


__global__ void Propagation(float* f, float* fout){
    
    __shared__ float* tile[qd*blockDim.y*blockDim.x]
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int w = gridDim.x*blockDim.x;
    int h = gridDim.y*blockDim.y;
    
    for(int idx=0; idx < blockDim.x; idx += blockDim.y){
        for(int n=0; n < qd; n++){
            tile[(n*h + (threadIdx.y+idx))*w + threadIdx.x] = fout[(n*h + (y+idx))*w + x]
        }
    }
    
    __syncthreads();
    
    f[(0*h + y)*w + x] = fout[(0*h + y)*w + x];
    f[(1*h + y)*w + (x+1)] = fout[(1*h + y)*w + x];
    f[(2*h + (y+1))*w + x] = fout[(2*h + y)*w + x];
    f[(3*h + y)*w + x] = fout[(3*h + y)*w + x];
    f[(4*h + y)*w + x] = fout[(4*h + y)*w + x];
    f[(5*h + y)*w + x] = fout[(5*h + y)*w + x];
    f[(6*h + y)*w + x] = fout[(6*h + y)*w + x];
    f[(7*h + y)*w + x] = fout[(7*h + y)*w + x];
    f[(8*h + y)*w + x] = fout[(8*h + y)*w + x];
    
}