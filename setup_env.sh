#!/bin/bash
# You will need root and evio to build the program
# Setup the path for these dependencies first

#source /home/chao/root-6.06.00/bin/thisroot.sh

if [ `uname -m` == 'x86_64' ]; then
    export THIRD_LIB=$PWD/thirdparty/lib64
else
    export THIRD_LIB=$PWD/thirdparty/lib
fi

export ET_LIB=/home/chao/PRad/coda/lib
export ET_INC=/home/chao/PRad/coda/include

export LD_LIBRARY_PATH=$THIRD_LIB:$ET_LIB:$LD_LIBRARY_PATH
