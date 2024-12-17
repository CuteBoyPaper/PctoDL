import sys
import os
import argparse
import time
import copy
import threading
import pynvml
import json
import torch
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import logging
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from tritonclient.utils import shared_memory as shm
from torchvision import transforms, datasets
import globalConfig
import dataLoader 
import doProfile 
import doInference 
import scheduler
from gevent import monkey
monkey.patch_all()

def find_bs(batch_size_list,tmp_bs):
    batch_size_list = batch_size_list[::-1]
    for bs in batch_size_list:
        if bs < tmp_bs:
            return bs
    return batch_size_list[0] 

def start_server(model):
    start_port = 8000
    gpu_resource_list = model.resources
    model_path = model.model_repository
    print(gpu_resource_list)
    index = 0
    start_time = time.time()
    os.system("sudo ./start_mps.sh")
    for i in range(len(gpu_resource_list)):
        gpu_time = time.time()
        if doInference.config_gpu(float(gpu_resource_list[i])) is False:
            print("Failed config gpu resource")
            return False
        gpu_time = time.time()-gpu_time
        print("gpu time: {:.3f} ms".format(gpu_time*1000))
        if doInference.create_docker_triton(start_port, start_port + 1, start_port + 2, model_path) is False:
            print("Failed to create TRITON server with config")
            return False
        start_port += 3
        index += 1
    end_time = time.time()
    print("start server cost time : {:.3f} ms".format((end_time-start_time)*1000))
    print("-------- ALL TRITON server are ready now. --------------------")

def batchdvfs_start(configs,log_txt):
    start_port = 8000
    model_list = [] 
    gpu_resource_list = configs.model.resources
    monitoer_executor = configs.monitor_executor
    model_executor = configs.model_executor
    for i in range(len(gpu_resource_list)):
        url = "localhost:"+str(start_port)
        client = httpclient.InferenceServerClient(url=url)
        model_list.append({"model_name":configs.model.model_name,"client":client,"data":None,"resource":gpu_resource_list[i],"batchsize":None})
        start_port += 3
    batchdvfs = scheduler.BatchDVFS(configs.model.power_cap,configs.model.alpha,configs.platform.sm_clocks,configs.model.batch_size_list,log_txt)
    log_txt.write("--------------CLIENT FINISHED------------------\n")
    batch_size_list = []
    tmp_bs = 1
    for i in range(len(gpu_resource_list)):
        batch_size_list.append(tmp_bs)
    frequency = 990
    memFreq = 9501
    total_throught = 0
    number = 0
    for i in range(100):
        log_txt.write(f"\n\nIteration {i+1}\t")
        log_txt.write("batchsize: {}\t frequency: {}\t memFreq: {}\n".format(batch_size_list[0],frequency,memFreq))
        for j in range(len(gpu_resource_list)):            
            data = dataLoader.load_images_to_batches(configs.model.data_path,batch_size_list[j])
            model_list[j]["batchsize"] = batch_size_list[j]
            model_list[j]["data"]=data
        tasks = []
        # 事件对象用于停止GPU信息采样
        stop_event = threading.Event()
        print("-------- Start testing. --------------------")
        # 启动GPU信息采样线程
        sampler_thread = monitoer_executor.submit(doProfile.sample_gpu_info,stop_event,configs.handle)
        # pynvml.nvmlDeviceSetMemoryLockedClocks(configs.handle,405,memFreq)
        pynvml.nvmlDeviceSetGpuLockedClocks(configs.handle,210,frequency)
        for j in range(len(model_list)):
            new_model_data = model_list[j]
            task = model_executor.submit(doInference.inference,new_model_data)
            tasks.append(task)
        
        wait(tasks, timeout=None, return_when=ALL_COMPLETED)
        # 任务完成后停止GPU信息采样
        stop_event.set()
        lantecy = 0
        throught = 0
        max_lantecy = 0
        for task in tasks:
            tmp_lantecy,tmp_throught = task.result()
            log_txt.write("inference time : {:.3f} ms Throught : {:.3f}\n".format(tmp_lantecy,tmp_throught))
            lantecy+=tmp_lantecy
            max_lantecy = max(tmp_lantecy,max_lantecy)
            throught+=tmp_throught
        
        lantecy = lantecy/len(tasks)
        log_txt.write("max inference time: {:.3f}ms, avg inference time : {:.3f} ms, total Throught : {:.3f}\n".format(max_lantecy,lantecy,throught))
        max_power,avg_power,max_gpu_utilization,avg_gpu_utilization,max_mem_utilization,avg_mem_utilization,max_pcie,avg_pcie_throught = sampler_thread.result()
        if i >=50 :
            total_throught+=throught
            number+=1
        log_txt.write("Max Power: {:.2f} W AVG Power: {:.2F} W\n".format(max_power,avg_power))
        log_txt.write("Max PCIE Throughput: {:.2f} MB/s  AVG PCIE Throughput: {:.2F} MB/s\n".format(max_pcie,avg_pcie_throught))
        log_txt.write("Max GPU Utilization: {:.2f}  AVG GPU Utilization: {:.2F}\n".format(max_gpu_utilization,avg_gpu_utilization))
        log_txt.write("Max Memory Bandwidth Utilization: {:.2f}  AVG Memory Bandwidth Utilization: {:.2F}\n".format(max_mem_utilization,avg_mem_utilization))
        frequency,next_bs = batchdvfs.scheduler(max_power,frequency,batch_size_list[0])
        for i in range(len(gpu_resource_list)):
            batch_size_list[i]=next_bs
        log_txt.flush()
        # print("-------- End testing. --------------------")
    log_txt.write("AVG Throughput: {:.2F} \n".format(total_throught/number))

def morak_start(configs,log_txt):
    start_port = 8000
    model_list = [] 
    gpu_resource_list = configs.model.resources
    monitoer_executor = configs.monitor_executor
    model_executor = configs.model_executor
    for i in range(len(gpu_resource_list)):
        url = "localhost:"+str(start_port)
        client = httpclient.InferenceServerClient(url=url)
        model_list.append({"model_name":configs.model.model_name,"client":client,"data":None,"resource":gpu_resource_list[i],"batchsize":None})
        start_port += 3
    morak = scheduler.Morak(configs.model.power_cap,configs.model.alpha,configs.model.slo,configs.model.belta,configs.platform.sm_clocks,configs.model.batch_size_list,log_txt)
    log_txt.write("--------------CLIENT FINISHED------------------\n")
    batch_size_list = []
    tmp_bs = 1
    for i in range(len(gpu_resource_list)):
        batch_size_list.append(tmp_bs)
    frequency = 990
    memFreq = 9501
    total_throught = 0
    number = 0
    for i in range(100):
        log_txt.write(f"\n\nIteration {i+1}\t")
        log_txt.write("batchsize: {}\t frequency: {}\t memFreq: {}\n".format(batch_size_list[0],frequency,memFreq))
        for j in range(len(gpu_resource_list)):            
            data = dataLoader.load_images_to_batches(configs.model.data_path,batch_size_list[j])
            model_list[j]["batchsize"] = batch_size_list[j]
            model_list[j]["data"]=data
        tasks = []
        # 事件对象用于停止GPU信息采样
        stop_event = threading.Event()
        print("-------- Start testing. --------------------")
        # 启动GPU信息采样线程
        sampler_thread = monitoer_executor.submit(doProfile.sample_gpu_info,stop_event,configs.handle)
        # pynvml.nvmlDeviceSetMemoryLockedClocks(configs.handle,405,memFreq)
        pynvml.nvmlDeviceSetGpuLockedClocks(configs.handle,frequency,frequency)
        for j in range(len(model_list)):
            new_model_data = model_list[j]
            task = model_executor.submit(doInference.inference,new_model_data)
            tasks.append(task)
        
        wait(tasks, timeout=None, return_when=ALL_COMPLETED)
        # 任务完成后停止GPU信息采样
        stop_event.set()
        lantecy = 0
        throught = 0
        max_lantecy = 0
        for task in tasks:
            tmp_lantecy,tmp_throught = task.result()
            log_txt.write("inference time : {:.3f} ms Throught : {:.3f}\n".format(tmp_lantecy,tmp_throught))
            lantecy+=tmp_lantecy
            max_lantecy = max(tmp_lantecy,max_lantecy)
            throught+=tmp_throught
        
        lantecy = lantecy/len(tasks)
        log_txt.write("max inference time: {:.3f}ms, avg inference time : {:.3f} ms, total Throught : {:.3f}\n".format(max_lantecy,lantecy,throught))
        max_power,avg_power,max_gpu_utilization,avg_gpu_utilization,max_mem_utilization,avg_mem_utilization,max_pcie,avg_pcie_throught = sampler_thread.result()
        if i >=50 :
            total_throught+=throught
            number+=1
        log_txt.write("Max Power: {:.2f} W AVG Power: {:.2F} W\n".format(max_power,avg_power))
        log_txt.write("Max PCIE Throughput: {:.2f} MB/s  AVG PCIE Throughput: {:.2F} MB/s\n".format(max_pcie,avg_pcie_throught))
        log_txt.write("Max GPU Utilization: {:.2f}  AVG GPU Utilization: {:.2F}\n".format(max_gpu_utilization,avg_gpu_utilization))
        log_txt.write("Max Memory Bandwidth Utilization: {:.2f}  AVG Memory Bandwidth Utilization: {:.2F}\n".format(max_mem_utilization,avg_mem_utilization))
        frequency,next_bs = morak.scheduler(max_lantecy,max_power,frequency,batch_size_list[0])
        for i in range(len(gpu_resource_list)):
            batch_size_list[i]=next_bs
        log_txt.flush()
    log_txt.write("AVG Throughput: {:.2F} \n".format(total_throught/number))
        # print("-------- End testing. --------------------")
def start(configs,log_txt):
    start_port = 8000
    model_list = [] 
    gpu_resource_list = configs.model.resources
    monitoer_executor = configs.monitor_executor
    model_executor = configs.model_executor
    for i in range(len(gpu_resource_list)):
        url = "localhost:"+str(start_port)
        client = httpclient.InferenceServerClient(url=url)
        model_list.append({"model_name":configs.model.model_name,"client":client,"data":None,"resource":gpu_resource_list[i],"batchsize":None})
        start_port += 3
    os.system("nvidia-smi -pl {}".format(configs.model.power_cap))
    sm_pid = scheduler.PID(configs.model.power_cap,configs.model.alpha,5e-5,1e-6,5e-7,configs.platform.sm_clocks,log_txt)
    mem_pid = scheduler.PID(configs.model.power_cap,configs.model.alpha,0,0,0,configs.platform.mem_clocks,log_txt)
    # bs_pid = scheduler.PID(configs.model.power_cap,configs.model.alpha,9e-5,7e-5,7e-5,configs.model.batch_size_list,log_txt)
    bs_pid = scheduler.PID(configs.model.power_cap,configs.model.alpha,0,0,0,configs.model.batch_size_list,log_txt)
    pid_scheduler = scheduler.PIDScheduler(sm_pid,mem_pid,bs_pid,log_txt)
    log_txt.write("--------------CLIENT FINISHED------------------\n")
    batch_size_list = configs.model.batch_size_list
    # tmp_bs = int(configs.model.max_bs/len(gpu_resource_list))
    # tmp_bs = 32
    # tmplist = [0,0,0,0]
    # for i in range(len(gpu_resource_list)):
    #     batch_size_list.append(tmp_bs+tmplist[i])
    print(batch_size_list)
    #默认为1050 1050
    frequency = 990
    memFreqList=[5001,9251,9501]
    memFreq = memFreqList[2]
    total_throught = 0
    number = 0
    time_2100=0
    for i in range(100):
        log_txt.write(f"\n\nIteration {i+1}\t")
        log_txt.write("batchsize: {}\t frequency: {}\t memFreq: {}\n".format(batch_size_list[0],frequency,memFreq))
        for j in range(len(gpu_resource_list)):            
            data = dataLoader.load_images_to_batches(configs.model.data_path,batch_size_list[j])
            model_list[j]["batchsize"] = batch_size_list[j]
            model_list[j]["data"]=data
        tasks = []
        # 事件对象用于停止GPU信息采样
        stop_event = threading.Event()
        print("-------- Start testing. --------------------")
        # 启动GPU信息采样线程
        sampler_thread = monitoer_executor.submit(doProfile.sample_gpu_info,stop_event,configs.handle)
        # pynvml.nvmlDeviceSetMemoryLockedClocks(configs.handle,memFreq,memFreq)
        pynvml.nvmlDeviceSetGpuLockedClocks(configs.handle,frequency,frequency)
        for j in range(len(model_list)):
            # new_model_data = copy.deepcopy(model_list[i])
            new_model_data = model_list[j]
            task = model_executor.submit(doInference.inference,new_model_data)
            tasks.append(task)
        
        wait(tasks, timeout=None, return_when=ALL_COMPLETED)
        # print("model : {} gpu resource: {} batchsize: {},frequency : {} HZ Memory Frequency : {:.3f}".format(configs.model.model_name,configs.model.resources,batch_size_list,frequency,memFreq))
        # 任务完成后停止GPU信息采样
        stop_event.set()
        lantecy = 0
        throught = 0
        for task in tasks:
            tmp_lantecy,tmp_throught = task.result()
            # print("inference time : {:.3f} ms Throught : {:.3f}".format(tmp_lantecy,tmp_throught))
            log_txt.write("inference time : {:.3f} ms Throught : {:.3f}\n".format(tmp_lantecy,tmp_throught))
            lantecy+=tmp_lantecy
            throught+=tmp_throught
        
        # print("avg inference time : {:.3f} ms Throught : {:.3f}".format(lantecy/len(tasks),throught))
        lantecy = lantecy/len(tasks)
        log_txt.write("avg inference time : {:.3f} ms total Throught : {:.3f}\n".format(lantecy,throught))
        max_power,avg_power,max_gpu_utilization,avg_gpu_utilization,max_mem_utilization,avg_mem_utilization,max_pcie,avg_pcie_throught = sampler_thread.result()
        if i >=50 :
            total_throught+=throught
            number+=1
        # print("Max Power: {:.2f} W AVG Power: {:.2F} W".format(max_power,avg_power))   
        # print("Max PCIE Throughput: {:.2f} MB/s  AVG PCIE Throughput: {:.2F} MB/s".format(max_pcie,avg_pcie_throught))
        # print("Max GPU Utilization: {:.2f}  AVG GPU Utilization: {:.2F}".format(max_gpu_utilization,avg_gpu_utilization)) 
        # print("Max Memory Bandwidth Utilization: {:.2f}  AVG Memory Bandwidth Utilization: {:.2F}".format(max_mem_utilization,avg_mem_utilization)) 
        log_txt.write("Max Power: {:.2f} W AVG Power: {:.2F} W\n".format(max_power,avg_power))
        log_txt.write("Max PCIE Throughput: {:.2f} MB/s  AVG PCIE Throughput: {:.2F} MB/s\n".format(max_pcie,avg_pcie_throught))
        log_txt.write("Max GPU Utilization: {:.2f}  AVG GPU Utilization: {:.2F}\n".format(max_gpu_utilization,avg_gpu_utilization))
        log_txt.write("Max Memory Bandwidth Utilization: {:.2f}  AVG Memory Bandwidth Utilization: {:.2F}\n".format(max_mem_utilization,avg_mem_utilization))
        frequency,memFreq,next_bs = pid_scheduler.scheduler(max_power,frequency,memFreq,batch_size_list[0])
        if frequency < 540 and memFreq!=5001:
            memFreq=memFreqList[max(memFreqList.index(memFreq)-1,0)]
            frequency = 660
        if frequency == 2100 and max_power/configs.model.power_cap<configs.model.alpha and memFreq!=9501:
            if time_2100>=1:
                memFreq=memFreqList[min(memFreqList.index(memFreq)+1,2)]
                frequency = 660
                time_2100=0
            time_2100+=1
        for i in range(len(gpu_resource_list)):
            batch_size_list[i]=next_bs
        log_txt.flush()
        # print("-------- End testing. --------------------")
    log_txt.write("AVG Throughput: {:.2F} \n".format(total_throught/number))


if __name__=="__main__":
    with open('./config/platform.json', 'r', encoding='utf-8') as file:
        platform_data = json.load(file)
    model_name = "vgg19"
    # algorithm_type = "morak"
    # algorithm_type = "batchdvfs"
    algorithm_type = "ours"
    if algorithm_type == "ours":
        with open('./config/vgg19/vgg19.json', 'r', encoding='utf-8') as file:
            model_data = json.load(file)
    elif algorithm_type == "morak":
        with open('./config/vgg19/vgg19_morak.json', 'r', encoding='utf-8') as file:
            model_data = json.load(file)
    elif algorithm_type == "batchdvfs":
        with open('./config/vgg19/vgg19_batchdvfs.json', 'r', encoding='utf-8') as file:
            model_data = json.load(file)
    configs = globalConfig.GlobalConfig(platform_data,model_data)
    with open(configs.model.log_path,"w") as log_txt:
        doInference.stop_all_docker()
        os.system("nvidia-smi -rmc")
        os.system("nvidia-smi -rgc")
        os.system("nvidia-smi -pl 250")
        os.system("sudo ./shutdown_mps.sh")
        start_server(configs.model)
        log_txt.write("--------------FINISH INITIAL------------------\n")
        if algorithm_type == "ours":
            start(configs,log_txt)
        elif algorithm_type == "morak":
            morak_start(configs,log_txt)
        elif algorithm_type == "batchdvfs":
            batchdvfs_start(configs,log_txt)
        doInference.stop_all_docker()
        os.system("nvidia-smi -rmc")
        os.system("nvidia-smi -rgc")
        os.system("sudo ./shutdown_mps.sh")