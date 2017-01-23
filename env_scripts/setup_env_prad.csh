#!/bin/csh

if (`uname -m` == 'x86_64') then
    setenv ET_LIB ${CODA}/Linux_x86_64/lib
    setenv ET_INC ${CODA}/common/include
    setenv THIRD_LIB  ${PWD}/thirdparty/lib64
else
    setenv ET_LIB ${CODA}/Linux_i686/lib
    setenv ET_INC ${CODA}/common/include
    setenv THIRD_LIB  ${PWD}/thirdparty/lib
endif

# comment this if you have your own root installed
setenv ROOTSYS /apps/root/5.34.21

if ! $?LD_LIBRARY_PATH then
	setenv LD_LIBRARY_PATH ${PWD}/decoder/libs:${THIRD_LIB}:${ET_LIB}:${ROOTSYS}/lib
else
	setenv LD_LIBRARY_PATH ${PWD}/decoder/libs:${THIRD_LIB}:${ET_LIB}:${ROOTSYS}/lib:${LD_LIBRARY_PATH}
endif
