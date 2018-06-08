#!/bin/bash

work_dir=$(cd $(dirname $0); pwd)
bin_dir=${work_dir}/bin
src_dir=${work_dir}/src
echo ${work_dir}
echo ${bin_dir}
echo ${src_dir}
pub_head=oclbin.h
itnl_head=internal.h
itnl_cpp=internal.cpp

rm -rf ${src_dir}/*.cpp
rm -rf ${src_dir}/*.h

echo "#ifndef _OCL_INTERNAL_HEAD_H" >  ${src_dir}/${itnl_head}
echo "#define _OCL_INTERNAL_HEAD_H" >> ${src_dir}/${itnl_head}
echo "" >> ${src_dir}/${itnl_head}
echo "#include \"${pub_head}\"" >> ${src_dir}/${itnl_head}
echo "" >> ${src_dir}/${itnl_head}

echo "#include \"${itnl_head}\"" > ${src_dir}/${itnl_cpp}
echo "#include \"${pub_head}\"" >> ${src_dir}/${itnl_cpp}
echo >> ${src_dir}/${itnl_cpp}
echo "extern const map<string, clKernel> allOCLKernels = {" >> ${src_dir}/${itnl_cpp}

num=$( ls -l "${bin_dir}" | grep '^-' | wc -l )
echo "num is ${num}"

for file in $( ls ${bin_dir} )
do
    if [ -d ${file} ]
    then
        echo ${file} is dir
    else
        var=$file
        var=${var//.*/}
        in_name=${bin_dir}/${file}
        out_name=${src_dir}/${var}.cpp
        echo ${in_name}
        echo ${out_name}

        ${work_dir}/bin2cpp -i ${in_name} -o ${out_name}
        echo "extern const size_t ${var}_len;" >> ${src_dir}/${itnl_head}
        echo "extern const uchar  ${var} [];" >> ${src_dir}/${itnl_head}
        echo "" >> ${src_dir}/${itnl_head}
        echo "  {\"${var}\", {"${var}", "${var}"_len}}," >> ${src_dir}/${itnl_cpp}
    fi
done
echo "};" >> ${src_dir}/${itnl_cpp}
echo "#endif" >>${src_dir}/${itnl_head}
