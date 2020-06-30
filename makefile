# MakeFile

# Python Scripts
PYDADOS = data_process.py

# Program Name
EXE = lbm

# Compilers
NVCC := nvcc
LINKER := g++

LIBRARY := yaml-cpp

# GPU
ARCH = sm_61

# Directories
SRCDIR := src
INCDIR := inc
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

OBJ := $(C_OBJ) $(CU_OBJ) $(CPP_OBJ)
DEP := $(C_DEP) $(CU_DEP) $(CPP_DEP)

# Flags
INCDIRS := $(addprefix -I, $(INCDIR))
CXXFLAGS := -std=c++11
NVCCARCHFLAG :=-arch $(ARCH)
NVCCFLAGS := -v --ptxas-options=-v -O3

INCYAML := $(pkg-config --cflags yaml-cpp)
LBPATH := $(pkg-config --libs yaml-cpp)

LIB := $(addprefix -l, $(LIBRARY))

.PHONY: all clean run

all: $(EXE)

# Linkage
$(EXE): $(OBJ)
	$(NVCC) $(NVCCARCHFLAG) $(CXXFLAGS) $^ -o $@ $(LIB)

# Generating Object files from .c
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(NVCC) -g $(INCDIRS) -c $< -o $@

# Generating Object files from .cu and .cpp
$(CU_OBJ): $(OBJDIR)/%.o: $(SRCDIR)/%.cu
$(CPP_OBJ): $(OBJDIR)/%.o: $(SRCDIR)/%.cpp
$(OBJ_COMP): | $(OBJDIR)
	$(NVCC) $(NVCCARCHFLAG) $(NVCCFLAGS) $(INCDIRS) --device-c $< -o $@ # Para debug flag -g -G e executar com cuda-memcheck ./lbm |more

# Generating Dependency files from .c
$(OBJDIR)/%.d: $(SRCDIR)/%.c | $(OBJDIR)
	$(NVCC) $(INCDIRS) -MM $< | sed -e 's%^%$@ %' -e 's% % $(OBJDIR)/%'\ > $@

# Generating Dependency files from .cu and .cpp
$(CU_OBJ): $(OBJDIR)/%.d: $(SRCDIR)/%.cu
$(CPP_OBJ): $(OBJDIR)/%.d: $(SRCDIR)/%.cpp
$(DEP_COMP): | $(OBJDIR)
	$(NVCC) $(INCDIRS) -MM $< | sed -e 's%^%$@ %' -e 's% % $(OBJDIR)/%'\ > $@

# Creating folder
$(OBJDIR):
	@mkdir -p $@

# Cleaning up
clean:
	@echo Cleaning up...
	@rm -f -r $(OBJDIR)
	@rm -f $(EXE)
	@rm -f -r $(VELOCITY)/*
	@rm -f -r $(RESULTS)/*
	@rm -f *.gif
	@echo Done!

# Running
run:
	@echo Running simulation...
	@./$(EXE)

	@echo Running Python Script...
	@python $(PYDADOS)

# Including dependency
-include $(DEP)