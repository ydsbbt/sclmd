.SUFFIXES:
.SUFFIXES: .f .F .o .a .f90 .F90 .c
#
#       interface to call siesta from python
#
default: pysiesta_socket
HOBJS=fsockets.o sockets.o fsiesta_sockets.o
HMOD=fsiesta.mod

HOBJSPIPE=fsiesta_pipes.o pxf.o 
HMODPIPE=fsiestapipe.mod

OBJS=siestaf.o  
#---------------------------------------------------------------------
#siesta src files
PYFC=gfortran #the fortran compiler used to compile siesta
SIESTADIR=/public/siesta/siesta-4.1.5-gcc-mpi
OBJDIR=Obj
#---------------------------------------------------------------------
VPATH:=$(SIESTADIR)/Src
ARCH_MAKE=$(SIESTADIR)/$(OBJDIR)/arch.make
include $(ARCH_MAKE)

CFLAGS=-fPIC

FC_DEFAULT:=$(FC)
FC_SERIAL?=$(FC_DEFAULT)
FC:=$(FC_SERIAL)         # Make it non-recursive

DEFS:=$(DEFS) $(DEFS_PREFIX)-UMPI $(DEFS_PREFIX)-UCDF
FPPFLAGS:=$(FPPFLAGS) $(DEFS_PREFIX)-UMPI $(DEFS_PREFIX)-UCDF

#FFLAGS= -g -traceback -check all -check nopointer -check noshape -warn all -DFC_HAVE_FLUSH -DFC_HAVE_ABORT -fPIC
#FFLAGS= -O3 -xW -DFC_HAVE_FLUSH -DFC_HAVE_ABORT -fPIC
FFLAGS= -O2  -DFC_HAVE_FLUSH -DFC_HAVE_ABORT -fPIC
#FORLIBS = -L/opt/intel/fce/10.1.015/lib/ -lguide -lifcore -lifport -limf
#FORLIBS = -L/opt/intel/fce/10.1.015/lib/ -lifcore -lifport -limf
FORLIBS = 

FSIESTA_OBJ=$(SIESTADIR)/Src/fsockets.o $(SIESTADIR)/Src/sockets.o $(SIESTADIR)/Src/fsiesta_sockets.o      # Client helper routines
FSIESTA_MOD=$(SIESTADIR)/Src/fsockets.mod $(SIESTADIR)/Src/sockets.mod $(SIESTADIR)/Src/fsiesta_sockets.mod    # Client helper module

#FSIESTA_OBJ==$(SIESTADIR)/Src/fsiesta_pipes.o      # Client helper routines
#FSIESTA_MOD=$(SIESTADIR)/Src/fsiesta_pipes.mod    # Client helper module

#PXF_OBJ=$(SIESTADIR)/Src/pxf.o              # Portable flush

pysiesta_socket: $(HOBJS) $(HMOD)
	f2py --f90exec=$(PYFC) --f90flags="$(FFLAGS)" $(FORLIBS) $(HOBJS) -c siestaf.f90 -m pysiesta_socket
	mkdir test
	cp pysiesta_socket*.so siestaf.py h2o/* test

#pysiesta_pipe: $(HOBJSPIPE) $(HMODPIPE)
#	f2py --f90exec=$(PYFC) --f90flags="$(FFLAGS)" $(FORLIBS) $(HOBJSPIPE) -c siestaf.f90 -m pysiesta_pipe
#	mkdir test
#	cp pysiesta_pipe*.so siestaf.py h2o/* test

clean: 
	@echo "--> Cleaning object and executable files"
	rm -r *.o *.mod pysiesta*.so test
