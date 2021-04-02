#!/bin/bash

filepath=$(cd "$(dirname "$0")"; pwd) 
workpath=${filepath}/../

echo ${filepath}
echo ${workpath}
cd ${workpath}

day=$(date "+%Y%m%d")
echo ${day}

release_folder=vasr.release.${day}
echo ${release_folder}
rm -rf ${release_folder}
rm -rf ${release_folder}.tar.gz
mkdir ${release_folder}

rm -rf build.armv7
mkdir build.armv7
cd build.armv7
../script/cmake_build_32.sh
make -j12; make install
mv install ../${release_folder}/armv7

cd ${workpath}
rm -rf build.aarch64
mkdir build.aarch64
cd build.aarch64
../script/cmake_build_64.sh
make -j12; make install
mv install ../${release_folder}/aarch64

cd ${workpath}
tar zcvf ${release_folder}.tar.gz ${release_folder}
