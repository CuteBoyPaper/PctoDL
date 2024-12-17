#!/usr/bin/env python3
import sys
import os
import argparse
import time
import threading
import pynvml
import torch
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import logging
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from tritonclient.utils import shared_memory as shm
from torchvision import transforms, datasets
sys.path.append(r"./data")
from imagenet import *
import torch.multiprocessing




torch.multiprocessing.set_sharing_strategy('file_system')
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

logger = logging.getLogger()



def get_success_inference_stats(json_data):
    """
    提取 JSON 数据中 `inference_stats` 字段下 `success` 中的 `count` 和 `ns` 变量。

    Args:
    - json_data (dict): JSON 格式的字典数据。

    Returns:
    - (success_count, success_ns): 元组，包含 `success` 中的 `count` 和 `ns` 值。
    """
    # 提取模型统计信息
    model_stats = json_data.get("model_stats", [])

    for model_stat in model_stats:
        inference_stats = model_stat.get("inference_stats", {})
        success_stats = inference_stats.get("success", {})
        success_count = success_stats.get("count", 0)
        success_ns = success_stats.get("ns", 0)
        
        return success_ns/1000000,success_count
    
    
    
def get_dataLoader(batchsize):
    data_transform = {
    "test": transforms.Compose([transforms.Resize([224,224]),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    test_dataset = TinyImageNet("./data/imagenet/tiny-imagenet-200", train=False,transform=data_transform["test"])   
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = int(batchsize), num_workers = 8, drop_last = True) 
    result = []
    for X,_ in test_loader:
        result.append(X.numpy())
    return result

def inference(model):
    model_name = model["model_name"]
    result=model["data"]
    resource=model["resource"]
    batchsize = int(model["batchsize"])
    client=model["client"]
    
    total_num = 0
    input_list = []
    output_list = []
    async_requests = []
    for input_data in result:
        inputs = httpclient.InferInput("input__0", input_data.shape, datatype="FP32")
        inputs.set_data_from_numpy(input_data, binary_data=True)
        outputs = httpclient.InferRequestedOutput(
            "output__0", binary_data=True, class_count=1000
        )
        input_list.append(inputs)
        output_list.append(outputs)
        total_num += input_data.shape[0] 
    result_list = []
    inference_statistics = client.get_inference_statistics(model_name)
    start_total_time,start_total_count =get_success_inference_stats(inference_statistics)
    for i in range(len(input_list)):
        client.infer(model_name=model_name, inputs=[input_list[i]]) 
    start = time.time()
    inference_statistics = client.get_inference_statistics(model_name)
    end_total_time,end_total_count = get_success_inference_stats(inference_statistics)
    end = time.time()
    print("time: {:.3f} ms".format((end-start)*1000))
    lantecy = (end_total_time-start_total_time)/(end_total_count-start_total_count)
    print("model : {} gpu resource: {}  batchsize: {}  inference time : {:.3f} ms Throught : {:.3f}".format(model_name,resource,batchsize,lantecy,batchsize*1000/lantecy))
    logger.info("model : {} gpu resource: {} batchsize: {}  inference time : {:.3f} ms Throught : {:.3f}".format(model_name,resource,batchsize,lantecy,batchsize*1000/lantecy))
    return lantecy,batchsize*1000/lantecy
def stop_all_docker():
    """
    before test or after test
    clear all active docker container
    """
    return os.system("docker stop $(docker ps -a -q)  >/dev/null 2>&1 ")


def create_docker_triton(port1, port2, port3, model_repository):
    """
    create a triton server
    port1: http port
    port2: grpc port
    model_repository: model repository path
    """
    cmd_create_triton_server = "docker run -d --gpus=1 --ipc=host --rm "
    cmd_create_triton_server += "-p{}:8000 -p{}:8001 -p{}:8002 ".format(
        port1, port2, port3)
    cmd_create_triton_server += "-v" + model_repository + ":/models "
    cmd_create_triton_server += "nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver "
    # cmd_create_triton_server += "--model-control-mode=explicit "
    cmd_create_triton_server += "--model-repository=/models "
    cmd_create_triton_server += "> /dev/null 2>&1 "
    os.system(cmd_create_triton_server)
    # logger.info(cmd_create_triton_server)

    # logger.debug("Waiting for TRITON Server to be ready at http://localhost:{}...".format(port1))
    live = "http://localhost:{}/v2/health/live".format(port1)
    ready = "http://localhost:{}/v2/health/ready".format(port1)
    count = 0
    while True:
        live_command = os.popen(
            "curl -i -m 1 -L -s -o /dev/null -w %{http_code} " + live).readlines()[0]
        ready_command = os.popen(
            "curl -i -m 1 -L -s -o /dev/null -w %{http_code} " + ready).readlines()[0]
        if live_command == "200" and ready_command == "200":
            # logger.debug("TRITON server is ready now. ")
            break
        # sleep for 1 s
        time.sleep(5)
        count += 1
        if count > 30:
            return False
    return True


def creat_docker_client(model, port, perf_file, sub_save_dir, input_data_dir, time_5s,batch):
    sub_save_dir = os.path.abspath(sub_save_dir)
    client_dir = os.path.abspath("./client")

    cmd_create_client = "docker run -d --rm --ipc=host --net=host "

    if input_data_dir is not None:
        input_data_dir = os.path.abspath(input_data_dir)
        cmd_create_client += "-v" + input_data_dir + ":/workspace/data "
    cmd_create_client += "-v" + client_dir + ":/workspace/myclient "
    cmd_create_client += "-v" + sub_save_dir + ":/workspace/sub_save_dir -w /workspace/sub_save_dir "
    cmd_create_client += "nvcr.io/nvidia/tritonserver:21.07-py3-sdk "
    cmd_create_client += "perf_analyzer -m {}  ".format(model)
    cmd_create_client += "-u localhost:{} -i grpc ".format(port)
    cmd_create_client += "-f {} ".format(perf_file)
    # cmd_create_client +="--measurement-mode=count_windows --measurement-request-count={} ".format(batch*10)
    cmd_create_client += "--shape input_0:3,224,224 "
    # cmd_create_client += "--request-distribution constant --request-rate-range {} ".format(
    #     rate)
    cmd_create_client += "-a --shared-memory system --max-threads 4 -v "
    cmd_create_client += "-r {} ".format(time_5s)
    cmd_create_client += "-b {} ".format(batch)

    if input_data_dir is not None:
        cmd_create_client += "--input-data ../data/{}.json ".format(model)
    # logger.info("client cmd : {}".format(cmd_create_client))
    os.system(cmd_create_client + " > /dev/null 2>&1 ")


def sample_gpu_info(stop_event):
    total_power = 0
    max_power = 0
    total_pcie_tx = 0
    max_pcie_tx = 0
    total_pcie_rx = 0
    max_pcie_rx = 0
    total_gpu_utilization = 0
    max_gpu_utilization = 0
    total_mem_utilization = 0
    max_mem_utilization = 0
    count = 0
    max_sm_activity_ratio = 0
    total_sm_activity_ratio = 0
    # max_sm_occupancy = 0
    # total_sm_occupancy = 0
    max_memory_bw_util = 0
    total_memory_bw_util = 0
   
    while not stop_event.is_set():
        
        #功率
        power_gpu=pynvml.nvmlDeviceGetPowerUsage(handle)/1000    
        total_power+=power_gpu
        max_power=max(max_power,power_gpu)
        # 获取 PCIe 吞吐量（发送和接收）
        pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024
        pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024
        total_pcie_rx+=pcie_rx
        max_pcie_rx=max(max_pcie_rx,pcie_rx)
        total_pcie_tx+=pcie_tx
        max_pcie_tx=max(max_pcie_tx,pcie_tx)
        # 获取 GPU 利用率
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        total_gpu_utilization+=utilization
        max_gpu_utilization=max(max_gpu_utilization,utilization)
        # 获取内存利用率(显存利用率)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # logger.info("memory used:{:.2f}   memory_total:{:.2f}".format(memory_info.used,memory_info.total))
        mem_uti = memory_info.used / memory_info.total * 100
        total_mem_utilization+=mem_uti
        max_mem_utilization=max(max_mem_utilization,mem_uti)
        count = count + 1
        time.sleep(0.005)
    
    logger.info("Max Power: {:.2f} W AVG Power: {:.2F} W".format(max_power,total_power/count))   
    logger.info("Max PCIE TX Throughput: {:.2f} MB/s  AVG PCIE TX Throughput: {:.2F} MB/s".format(max_pcie_tx,total_pcie_tx/count))
    logger.info("Max PCIe RX Throughput: {:.2f} MB/s AVG PCIe RX Throughput: {:.2F} MB/s".format(max_pcie_rx,total_pcie_rx/count))
    logger.info("Max GPU Utilization: {:.2f}  AVG GPU Utilization: {:.2F}".format(max_gpu_utilization,total_gpu_utilization/count)) 
    logger.info("Max Memory Bandwidth Utilization: {:.2f}  AVG Memory Bandwidth Utilization: {:.2F}".format(max_mem_utilization,total_mem_utilization/count)) 
    return max_power,total_gpu_utilization/count,total_mem_utilization/count,(total_pcie_rx+total_pcie_tx)/count
    


def test_models(model_config_list, repository_path, save_dir, input_data, time_5s,executor,monitoer_executor):
    """
    test models 
    parallel running multiply models with different TRITON server in one host
    """
    repository_path = os.path.abspath(repository_path)
    save_dir = os.path.abspath(save_dir)
    start_port = 8000
    index = 1
    # models_path = os.path.join(repository_path, "model")
    start_time = time.time()
    stop_all_docker()
    for i in range(len(model_config_list)):
        model, resource, batch = model_config_list[i].split(":")
        model_path = repository_path
        batch_time = time.time()
        if config_batch(model_path, model, batch) is False:
            logger.error("Failed to config batch size: " + model + " - " + batch)
            return False
            # config gpu failed, return
        batch_time = time.time()-batch_time
        # logger.info("batch time: {:.3f} ms".format(batch_time*1000))
        gpu_time = time.time()
        if config_gpu(float(resource), model_path) is False:
            logger.error("Failed config gpu resource: " + resource)
            return False
        gpu_time = time.time()-gpu_time
        # logger.info("gpu time: {:.3f} ms".format(gpu_time*1000))
        if create_docker_triton(start_port, start_port + 1, start_port + 2, model_path) is False:
            logger.error("Failed to create TRITON server with config: " + model_config_list[i])
            return False
        start_port += 3
        index += 1
    end_time = time.time()
    logger.info("start server cost time : {:.3f} ms".format((end_time-start_time)*1000))
    logger.info("-------- ALL TRITON server are ready now. --------------------")
    # sys.exit(0)
    
    start_port = 8000
    model_list = [] 
    for i in range(len(model_config_list)):
        model, resource, batch = model_config_list[i].split(":")
        data = get_dataLoader(batch)
        url = "localhost:"+str(start_port)
        client = httpclient.InferenceServerClient(url=url)
        model_list.append({"model_name":model,"client":client,"data":data,"resource":resource,"batchsize":batch})
        start_port += 3
    logger.info("-------- Start testing. --------------------")
    index = 1
    tasks = []
    # 事件对象用于停止GPU信息采样
    stop_event = threading.Event()
    
    # 启动GPU信息采样线程
    # sampler_thread = threading.Thread(target=sample_gpu_info, args=(stop_event,))
    sampler_thread = monitoer_executor.submit(sample_gpu_info,stop_event)
    # sampler_thread.start()
    for i in range(len(model_list)):
        task = executor.submit(inference,model_list[i])
        # inference(model_list[i])
        index += 1
        tasks.append(task)
    # time.sleep(10)
    # while task.done()==False:
    #             power_gpu=pynvml.nvmlDeviceGetPowerUsage(glva.handle)/1000
    #             time.sleep(0.001)
    wait(tasks, timeout=None, return_when=ALL_COMPLETED)
    # 任务完成后停止GPU信息采样
    stop_event.set()
    lantecy = 0
    throught = 0
    for task in tasks:
        tmp_lantecy,tmp_throught = task.result()
        lantecy+=tmp_lantecy
        throught+=tmp_throught
    logger.info("avg inference time : {:.3f} ms total Throught : {:.3f}".format(lantecy/len(tasks),throught))
    max_power,gpu_utilization,mem_utilization,pcie_throught = sampler_thread.result()
    logger.info("Max Power: {:.2f} W ".format(max_power))   
    logger.info("AVG PCIE Throughput: {:.2F} MB/s".format(pcie_throught))
    logger.info("AVG GPU Utilization: {:.2F}".format(gpu_utilization)) 
    logger.info("AVG Memory Bandwidth Utilization: {:.2F}".format(mem_utilization))
    # sampler_thread.join()
    logger.info("-------- End testing. --------------------")



def config_gpu(gpu_resource, model_repository):
    """
    config gpu resource before each triton starting
    """
    os.system("sudo nvidia-cuda-mps-control -d " + "> /dev/null 2>&1 ")
    server_id = os.popen("echo get_server_list | nvidia-cuda-mps-control").readlines()
    # logger.info(server_id)
    if len(server_id) == 0:
        # no mps server is runnning
        if create_docker_triton(8000, 8001, 8002, model_repository) is False:
            logger.error("Start triton server failed in config gpu time. ")
            return False
        stop_all_docker()
        server_id = os.popen("echo get_server_list | nvidia-cuda-mps-control").readlines()[0].strip('\n')
    else:
        server_id = server_id[0].strip('\n')

    gpu_set_cmd = "echo set_active_thread_percentage {} {} | sudo nvidia-cuda-mps-control".format(server_id,
                                                                                                  gpu_resource)
    gpu_resource_set = os.popen(gpu_set_cmd).readlines()[0].strip('\n')

    if float(gpu_resource_set) != gpu_resource:
        logger.error("Failed to config gpu resource")
        return False
    else:
        return True
        # logger.info("Success to set gpu resource: {}".format(float(gpu_resource_set)))
    return True


def config_batch(model_path, model_name, batch_size):
    """
    config inference batch size before each triton starting
    """
    config_file_path = os.path.join(model_path, model_name)
    config_file_path = os.path.join(config_file_path, "config.pbtxt")
    # logger.info("config file path : {}".format(config_file_path))
    if os.path.isfile(config_file_path) is False:
        logger.error("config file path : {}".format(config_file_path))
        logger.error("{} config not existed! ".format(model_name))
        return False
    else:
        return True


if __name__ == "__main__":
    os.system("nvidia-smi -rmc")
    os.system("nvidia-smi -rgc")
    
    logger.info("-------- Program test_inference.py is starting. ---------------")
    

    modelName="densenet201"
    batchsize = 64
    frequency = 2100
    memFreq = 9501
    repository_path="./model_repositorys/"+modelName+"_repository/"
    
    filename = modelName + "_morak.txt"
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=filename,
                        filemode='w')
    model_config_lists = []
    
    resourceNum = [1,2,3,4,5]
    input_data = None
    time_5s = 1
    save_dir = "./perf_data/"
    if input_data is None:
        logger.info("Using random data for inference. ")
    else:
        logger.info("Using real data for inference. ")
    
    stop_all_docker()
    monitoer_executor = ThreadPoolExecutor(max_workers=1)

    executor = ThreadPoolExecutor(max_workers=1)
    gpu_resource_list = [10,20,30,40,50,60,70,80,90,100]
    # gpu_resource_list = [100]
    for gpu_resource in gpu_resource_list:
        model_config = modelName+":"+str(gpu_resource)+":"+ str(batchsize)  
        model_config_list = [model_config]
        print(model_config_list)
        test_models(model_config_list, repository_path, save_dir, input_data, time_5s,executor,monitoer_executor)
        time.sleep(10)
    
    stop_all_docker()
    # 销毁 GPU 组并关闭 DCGM
    os.system("sudo ./shutdown_mps.sh")
    
    logger.info("-------- Program test_inference.py stopped. --------------------")