#!/bin/bash
#!python
#先启动多个容器while true共同运行，再单独启动一个pytorch进程，进行测试
batchsizeList=(1 2 4 8 16 32 64)
# modelNameList=("densenet201" "resnet50" "mobilenetv2" "vgg19" "convnext_base" "maxvit" "efficientnet_v2_m" "swin_b")
# batchsizeList=(2)
modelNameList=("convnext_base" "maxvit" "efficientnet_v2_m" "swin_b")
analysisTypeList=("ncu" "nsys")
# analysisTypeList=("ncu")
for modelName in ${modelNameList[@]};
do
    for batchsize in ${batchsizeList[@]};
    do
        for analysisType in ${analysisTypeList[@]};
        do
            if [ "$analysisType" = "ncu" ]; then
                echo "tianxueyang" | sudo -S which
                echo "start ncu"
                sudo ncu --csv --set detailed --nvtx --nvtx-include "start/" --replay-mode application  python test.py --model_name  ${modelName} --batchsize ${batchsize} --type ${analysisType}  > ./result/${modelName}/${modelName}_${batchsize}/output_ncu.csv
                sleep 1
                sudo /usr/local/NVIDIA-Nsight-Compute/ncu -o ./result/${modelName}/${modelName}_${batchsize}/output_ncu -f --set detailed --replay-mode application --nvtx --nvtx-include "start/" python test.py --model_name  ${modelName} --batchsize ${batchsize} --type ${analysisType}
                # echo "sudo /usr/local/cuda/bin/ncu -o ./result/${modelName}/${modelName}_${batchsize}/output_ncu -f --set detailed --replay-mode application --nvtx --nvtx-include "start/" python test.py --model_name  ${modelName} --batchsize ${batchsize} --type ${analysisType}"
                sleep 1
            else
                echo "tianxueyang" | sudo -S which
                echo "start nsys"
                sudo /opt/nvidia/nsight-systems/2024.4.1/bin/nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s process-tree -o ./result/${modelName}/${modelName}_${batchsize}/output_nsys --cudabacktrace=true --capture-range=cudaProfilerApi --capture-range-end=stop  -f true -x true python test.py --model_name  ${modelName} --batchsize ${batchsize} --type ${analysisType}
                
                sleep 1
                sudo /opt/nvidia/nsight-systems/2024.4.1/bin/nsys stats --force-export=true --report gputrace --format csv,column --output .,- ./result/${modelName}/${modelName}_${batchsize}/output_nsys.nsys-rep
                
                sleep 1
            fi
        done
    done
done