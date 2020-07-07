# MakeFile

# Python Scripts
PYMESH = Python/mesh.py
PYDADOS = Python/data_process.py

# Program Name
EXE = lbm

# Compilers
NVCC := nvcc

LIBRARY := yaml-cpp
LIBDIR := /usr/local/lib

# GPU
ARCH = sm_61

# Directories
SRCDIR := src
INCDIR := inc /usr/local/include
OBJDIR := obj
BINDIR := bin
VELOCITY := Velocity
RESULTS := Results

# Files
C_FILES := $(wildcard $(SRCDIR)/*.c)
CU_FILES := $(wildcard $(SRCDIR)/*.cu)
CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)

C_OBJ := $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(C_FILES))
CU_OBJ := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CU_FILES))
CPP_OBJ := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CPP_FILES))

C_DEP := $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.d, $(C_FILES))
CU_DEP := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.d, $(CU_FILES))
CPP_DEP := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.d, $(CPP_FILES))

OBJ_COMP := $(CU_OBJ) $(CPP_OBJ)
DEP_COMP := $(CU_DEP) $(CPP_DEP)

OBJ := $(C_OBJ) $(CPP_OBJ) $(CU_OBJ)
DEP := $(C_DEP) $(CPP_DEP) $(CU_DEP)

# Flags
INCDIRS = $(addprefix -I, $(INCDIR))
LIBDIRS = $(addprefix -L, $(LIBDIR))
LIB := $(addprefix -l, $(LIBRARY))
CXXFLAGS := -std=c++11
NVCCARCHFLAG :=-arch $(ARCH)
NVCCFLAGS := -v --ptxas-options=-v -O3 --device-c # Para debug flag -g -G e executar com cuda-memcheck ./lbm |more
DEPFLAGS := -MMD

COMPILE.c = $(NVCC) $(DEPFLAGS) -g $(INCDIRS) -c
COMPILE.cpp = $(NVCC) $(DEPFLAGS) $(CXXFLAGS) $(INCDIRS) $(NVCCARCHFLAG) $(NVCCFLAGS)

.PHONY: all clean run

all: $(EXE)

# Linkage
$(EXE): $(OBJ)
	$(NVCC) $(NVCCARCHFLAG) $(CXXFLAGS) $(LIBDIRS) $^ -o $@ $(LIB)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(COMPILE.c) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(COMPILE.cpp) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(COMPILE.cpp) $< -o $@

$(OBJDIR):
	@mkdir -p $@

# Cleaning up
clean:
	@echo Cleaning up...
	@rm -f -r $(OBJDIR)
	@rm -f -r $(DEPDIR)
	@rm -f $(EXE)
	@rm -f -r $(VELOCITY)/*
	@rm -f -r $(RESULTS)/*
	@rm -f *.gif
	@echo Done!

# Running
run:
	@echo Generating mesh...
	@python $(PYMESH)
	
	@echo Running simulation...
	@./$(EXE)

	@echo Running Python Script...
	@python $(PYDADOS)

# Including dependency
-include $(DEP)