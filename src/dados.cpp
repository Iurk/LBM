#include <iostream>
#include "yaml-cpp/yaml.h"

#include "utilidades.h"
#include "dados.h"
#include "paths.h"

// Opening yaml file
YAML::Node config = YAML::LoadFile("./bin/dados.yml");
YAML::Node config_lattice = YAML::LoadFile("./bin/lattices.yml");

// Getting sections
const YAML::Node& domain = config["domain"];
const YAML::Node& simulation = config["simulation"];
const YAML::Node& gpu = config["gpu"];
const YAML::Node& input = config["input"];

std::string Lattice = simulation["lattice"].as<std::string>();
const YAML::Node& lattice = config_lattice[Lattice];

namespace myGlobals{

	//Domain
	unsigned int Nx = domain["Nx"].as<int>();
	unsigned int Ny = domain["Ny"].as<int>();
	unsigned int D = domain["D"].as<int>();

	//Simulation
	unsigned int NSTEPS = simulation["NSTEPS"].as<int>();
	unsigned int NSAVE = simulation["NSAVE"].as<int>();
	unsigned int NMSG = simulation["NMSG"].as<int>();
	bool meshprint = simulation["meshprint"].as<bool>();
	double erro_max = simulation["erro_max"].as<double>();

	//GPU
	unsigned int nThreads = gpu["nThreads"].as<unsigned int>();

	//Input
	double u_max = input["u_max"].as<double>();
	double rho0 = input["rho0"].as<double>();
	double Re = input["Re"].as<double>();

	//Lattice Info
	unsigned int ndir = lattice["q"].as<unsigned int>();
	std::vector<int> ex_vec = lattice["ex"].as<std::vector<int>>();
	std::vector<int> ey_vec = lattice["ey"].as<std::vector<int>>();
	std::string cs_str = lattice["cs"].as<std::string>();
	std::string w0_str = lattice["w0"].as<std::string>();
	std::string wp_str = lattice["wp"].as<std::string>();
	std::string ws_str = lattice["ws"].as<std::string>();

	int *ex = ex_vec.data();
	int *ey = ey_vec.data();
	double cs = equation_parser(cs_str);
	double w0 = equation_parser(w0_str);
	double wp = equation_parser(wp_str);
	double ws = equation_parser(ws_str);

	//Memory Sizes
	const size_t mem_mesh = sizeof(bool)*Nx*Ny;
	const size_t mem_size_ndir = sizeof(double)*Nx*Ny*ndir;
	const size_t mem_size_scalar = sizeof(double)*Nx*Ny;

	// Nu and Tau
	double nu = (u_max*D)/Re;
	const double tau = nu/(cs*cs) + 0.5;

	bool *cylinder = read_bin(solid_mesh);
	bool *fluid = read_bin(fluid_mesh);
}