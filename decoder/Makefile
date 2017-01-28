################################################################################
# Makefile for building: PRad analysis library and some example programs       #
################################################################################

MAKEFILE      = Makefile
PRAD_PATH     = ..
PRAD_LIBS     = libs

####### Compiler, tools and options
CC            = gcc
CXX           = g++
FORTRAN       = gfortran
FFLAGS        = -fPIC
CXXFLAGS      = -std=c++11 -O1 -g -pipe -Wall -m64 -mtune=generic -fPIC \
				-fstack-protector-strong --param=ssp-buffer-size=4 \
				-grecord-gcc-switches -mtune=generic -fPIC
INCPATH       = -Iinclude -I$(PRAD_PATH)/include -I$(PRAD_LIBS)/include \
				-I$(ROOTSYS)/include
DEL_FILE      = rm -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p
COPY          = cp -f
COPY_FILE     = cp -f
COPY_DIR      = cp -f -R
INSTALL_FILE  = install -m 644 -p
INSTALL_PROGRAM = install -m 755 -p
INSTALL_DIR   = cp -f -R
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
TAR           = tar -cf
COMPRESS      = gzip -9f
LINK          = g++
LFLAGS_LIBS   = -shared -Wl,-O1 -Wl,-z,relro
LFLAGS        = -Wl,-O1 -Wl,-z,relro
LIBS          = -lpthread -lexpat \
                -L$(PRAD_LIBS) -lprana -lcneural \
				-L$(ROOTSYS)/lib -lCore -lRint -lRIO -lNet -lHist \
				-lGraf -lGraf3d -lGpad -lTree -lPostscript -lMatrix \
				-lPhysics -lMathCore -lThread -lGui -lSpectrum
AR            = ar cqs
RANLIB        = 
SED           = sed
STRIP         = 

####### Targets, add new objects here

EXAMPLE_SRC   = src
EXAMPLE_DIR   = .
EXAMPLE       = testGEM \
                testHyCal \
				testDST \
                testLMS \
                testEPICS \
                testMatch \
                testSim \
				testPerform \
                getAvgGain \
                replay \
                eventSelect \
                beamChargeCount \
				cosmicCheck \
				mollerCenter \
				physCalib \
				testNeural \
				neuralTrain \
				neuralReject

EXAMPLE_BINS  = $(addprefix $(EXAMPLE_DIR)/, $(EXAMPLE))


####### Build rules
first: all

all: lib exe

.PHONY: lib cleanobj cleanlib

lib:
	$(MAKE) -C $(PRAD_LIBS)
#cd $(PRAD_LIBS) && $(MAKE)

exe:
	 $(MAKE) $(EXAMPLE_BINS)

$(EXAMPLE_DIR)/%: $(EXAMPLE_SRC)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(INCPATH) $(LIBS)

####### Clean
clean: cleanobj cleanexe cleanlib

cleanexe:
	$(DEL_FILE) $(EXAMPLE)

cleanobj:
	$(MAKE) -C $(PRAD_LIBS) cleanobj

cleanlib:
	$(MAKE) -C $(PRAD_LIBS) cleanlib

####### Install

install:   FORCE

uninstall:   FORCE

FORCE:

