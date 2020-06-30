#include <iostream>
#include <string.h>

#include "tinyexpr.h"
#include "utilidades.h"

double equation_parser(const std::string equation){

	char expr[32];

	strcpy(expr, equation.c_str());

	double result = te_interp(expr, 0);

	return result;
}