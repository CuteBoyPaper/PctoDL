import sys
import os
import argparse
import time
import json
import torch
import pynvml
import threading
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import logging
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from tritonclient.utils import shared_memory as shm
from torchvision import transforms, datasets
from pytriton.client import ModelClient
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(r"./data")
from imagenet import *
from PIL import Image
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename='test_inference.log',
                        filemode='w')
logger = logging.getLogger()
def load_images_to_batches(image_dir, batch_size):
    """
    从指定文件夹中读取图像，并按指定的 batch_size 组织成 [batchsize, 3, 224, 224] 的数组。
    仅返回完整的批次。

    :param image_dir: str, 图片文件夹路径
    :param batch_size: int, 每个批次的大小
    :return: list, 包含每个完整批次的 NumPy 数组列表
    """
    # 初始化保存批次数据的列表
    batches = []

    # 获取目录中的所有图片文件
    image_files = [file for file in os.listdir(image_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # 初始化一个临时列表，用于保存当前批次的图片数据
    current_batch = []

    for image_file in image_files:
        # 构建图片的完整路径
        image_path = os.path.join(image_dir, image_file)

        try:
            # 打开图片
            with Image.open(image_path) as img:
                # 将图片调整为 (224, 224) 尺寸
                img = img.resize((224, 224))
                
                # 转换为 NumPy 数组
                img_array = np.array(img, dtype=np.float32)
                
                # 如果图片是灰度图像或单通道，转换为 RGB
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)

                # 确保图像是 RGB 的
                if img_array.shape[2] != 3:
                    raise ValueError(f"Image {image_file} is not in RGB format.")

                # 转换为 [3, 224, 224] 形状
                img_array = img_array.transpose((2, 0, 1))
                
                # 将当前图片添加到批次
                current_batch.append(img_array)

                # 如果当前批次达到 batch_size，则保存批次并重置
                if len(current_batch) == batch_size:
                    batches.append(np.stack(current_batch))
                    current_batch = []

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # 如果最后一个批次不满 batch_size，则舍弃该批次
    return batches
                        
def get_dataLoader(batchsize):
    data_transform = {
    "test": transforms.Compose([transforms.Resize([224,224]),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    test_dataset = TinyImageNet("./data/imagenet/tiny-imagenet-200", train=False,transform=data_transform["test"])   
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = int(batchsize), num_workers = 8, drop_last = True) 
    # result = []
    # for X,_ in test_loader:
    #     result.append(X.numpy())
    return test_loader


def creat_model(model_path,model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """
        create model
    """
    net = None
    match model_name:
        case "densenet201":
            net=models.densenet201()
        case "alexnet":
            net=models.alexnet()
        case "resnet50":
            net=models.resnet50()
        case "mobilenetv2":
            net=models.mobilenet_v2()
        case "vgg19": 
            net=models.vgg19()
        case "convnext_base":
            net=models.convnext_base()
        case "maxvit":
            net=models.maxvit_t()
        case "efficientnet_v2_m":
            net=models.efficientnet_v2_m()
        case "swin_b":
            net=models.swin_b()
    model_path = os.path.join(model_path,model_name+".pth")
    net.load_state_dict(torch.load(model_path), strict=False)#保存的训练模型
    net.to("cuda:0")
    net.eval()#切换到eval（）
    return net

def warmup(model,batchsize):
    batch_size = int(batchsize)
    data_loader = get_dataLoader(batchsize)
    total_num = 0
    i=0
    start = time.time()
    with torch.no_grad():
        for X,Y in data_loader:
            X = X.to("cuda:0")
            # torch.cuda.nvtx.range_push("start")
            # torch.cuda.profiler.cudart().cudaProfilerStart()
            model(X)
            # torch.cuda.nvtx.range_pop()
            # torch.cuda.profiler.cudart().cudaProfilerStop()
            total_num +=X.shape[0]
            i+=1
            if i==5:
                break
    end = time.time()
    spend = end - start   
    # print("inference time : {:.3f} ms Throught : {:.3f}".format(batchsize*1000*spend/total_num,total_num/(spend)))
def inferencePytorch(model,batchsize,data_loader):
    batch_size = int(batchsize)
    total_num = 0
    i=0
    total_time = 0
    with torch.inference_mode():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for X,Y in data_loader:
            X = X.to("cuda:0")
            start_event.record()
            model(X)
            end_event.record()
            end_event.synchronize()
            total_num +=X.shape[0]
            total_time+=start_event.elapsed_time(end_event)
            i+=1
            if i==10:
                break
    spend = total_time/i
    
    return batchsize*1000/spend,total_time/i
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
    return max_power,total_gpu_utilization/count,total_mem_utilization/count,(total_pcie_rx+total_pcie_tx)/count
    

if __name__ == "__main__":
    # os.system("docker stop $(docker ps -a -q)  >/dev/null 2>&1 ")
    # batchsizeList = [args.batchsize]
    model_name = "densenet201"
    frequencylist=[1950,1830,1710,1590,1470,1350,1230,1110,990,870,750,630,510,390,270] # 3080Ti    
    # frequencylist=[2100, 1980, 1860]
    memFrequencylist = [9501,9251,5001,810]
    # memFrequencylist = [9501,9251,5001]
    # frequencylist.reverse()
    # memFrequencylist.reverse()
    model = creat_model("./models",model_name)
    batchsizeList = [128]
    # os.system("./shutdown_mps.sh")
    os.system("nvidia-smi -rmc")
    os.system("nvidia-smi -rgc")
    # data_loader = get_dataLoader(batchsize)
    print("NVIDIA TRITION :")
    warmup(model,batchsizeList[0])
    warmup(model,batchsizeList[0])
    print("model: {}".format(model_name))
    pynvml.nvmlDeviceResetGpuLockedClocks(handle)
    pynvml.nvmlDeviceSetPowerManagementLimit(handle,pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle))
    pynvml.nvmlDeviceResetMemoryLockedClocks(handle)
    monitoer_executor = ThreadPoolExecutor(max_workers=1)
    executor = ThreadPoolExecutor(max_workers=1)
    time.sleep(1)
    for batch_size in batchsizeList:
        data_loader = get_dataLoader(batch_size)
        for memFrequency in memFrequencylist:
            # cmd = "nvidia-smi -lmc "+memFrequency+""
            # os.system("")
            pynvml.nvmlDeviceSetMemoryLockedClocks(handle,memFrequency,memFrequency)
            for frequency in frequencylist:
                pynvml.nvmlDeviceSetGpuLockedClocks(handle,frequency,frequency)
                time.sleep(1)
            
                exectime = 0
                throught = 0
                max_power = 0
                gpu_utilization = 0
                mem_utilization = 0
                pcie_throught = 0
                n = 10
                for i in range(n):
                    # 事件对象用于停止GPU信息采样
                    stop_event = threading.Event()
                    # 启动GPU信息采样线程
                    # sampler_thread = threading.Thread(target=sample_gpu_info, args=(stop_event,))
                    sampler_thread = monitoer_executor.submit(sample_gpu_info,stop_event)
                    task = executor.submit(inferencePytorch,model,batch_size,data_loader)
                    wait([task], timeout=None, return_when=ALL_COMPLETED)
                    # 任务完成后停止GPU信息采样
                    stop_event.set()
                    tmp_throught,tmp_exectime= task.result()
                    exectime+=tmp_exectime
                    throught+=tmp_throught
                    tmp_max_power,tmp_gpu_utilization,tmp_mem_utilization,tmp_pcie_throught = sampler_thread.result()
                    max_power+=tmp_max_power
                    gpu_utilization+=tmp_gpu_utilization
                    mem_utilization+=tmp_mem_utilization
                    pcie_throught+=tmp_pcie_throught
                    time.sleep(1)
                print("model :{} ,batchsize :{},frequency :{},memFreq :{},inference time :{:.3f} ms,Throught :{:.3f},max Power :{:.3f} W,GPU util :{:.3f}%,Mem util :{:.3f}%,Pcie throught :{:.3f}".format(model_name,batch_size,frequency,memFrequency,exectime/n,throught/n,max_power/n,gpu_utilization/n,mem_utilization/n,pcie_throught/n))