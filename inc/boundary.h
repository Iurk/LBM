#ifndef __BOUNDARY_H
#define __BOUNDARY_H

void inlet_BC(double, double*, double*, double*, double*);
void outlet_BC(double*);
void bounce_back(double*);
void noslip(double *);

#endif