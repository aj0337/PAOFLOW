# extlibs/install/make_iotk.inc.  Generated from make_iotk.inc.in by configure.

# -----------------------------
# compilation rules

.SUFFIXES :
.SUFFIXES : .o .c .f .f90

.f90.o:
	$(MPIF90) $(F90FLAGS) -c $<

.f.o:
	$(F77) $(FFLAGS) -c $<

.c.o:
	$(CC) $(CFLAGS) -c $<

# -----------------------------
# compiler definitions


CC                  = icc
CFLAGS              = -O3 $(DFLAGS) $(IFLAGS)
CPP                 = cpp
CPPFLAGS            = -P -traditional $(DFLAGS) $(IFLAGS)
F90                 = ifort
MPIF90              = mpiifort
F90FLAGS            = $(FFLAGS) -nomodule -fpp $(FDFLAGS) $(IFLAGS) 
F77                 = ifort
FFLAGS              = -O2 -assume byterecl -g -traceback
FFLAGS_NOOPT        = -O0 -assume byterecl -g -traceback
LD                  = mpiifort
LDFLAGS             = 
AR                  = ar
ARFLAGS             = ruv
RANLIB              = ranlib


#------------------------------
# specific definitions

DFLAGS         = -D__INTEL -D__FFTW3 -D__MPI -D__PARA
FDFLAGS        = $(DFLAGS)
IFLAGS         = $(AUTO_INCLUDE)
LIBOBJS        = $(AUTO_LIBS)


# -----------------------------
# libraries

TOPDIR              = /home/anooja/PAOFLOW_quantran/src/externals/transportPAO
#
LIB_MPI             = 
LIB_MASS            = 
#
LIBS                = $(LIB_MPI) $(LIB_MASS)

