--------------------------------------------------------------------
WanT can take advantage of several optimized numerical libraries
(essl, fftw, mkl...).  This configure script attempts to find them,
but may fail if they have been installed in non-standard locations.

The following libraries have been found:

  BLAS_LIBS=-L/opt/intel/oneapi/mkl/2022.1.0/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
  LAPACK_LIBS=-L/opt/intel/oneapi/mkl/2022.1.0/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
  FFT_LIBS= -L/opt/intel/oneapi/mkl/2022.1.0/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
  IOTK_LIBS=-L/home/anooja/PAOFLOW_quantran/src/externals/transportPAO/extlibs/iotk/lib -liotk
  @delete@
  @delete@
  @delete@
  @delete@
  @delete@
  @delete@
  @delete@

Please check if that is correct.

If any libraries are missing, you may specify a list of directories
to search and retry, as follows:
  ./configure LIBDIRS="list of directories, separated by spaces"

For more info, read the WanT Manual (docs/want_manual.pdf).
--------------------------------------------------------------------
