import sys
import os
import argparse
import time
import json
import torch
import numpy as np
from torchvision import models
from torchvision import transforms, datasets
import logging
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(r"../data")
from imagenet import *
from PIL import Image
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
    test_dataset = TinyImageNet("../data/imagenet/tiny-imagenet-200", train=False,transform=data_transform["test"])   
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
    print("inference time : {:.3f} ms Throught : {:.3f}".format(batchsize*1000*spend/total_num,total_num/(spend)))
def inferencePytorchWithNcu(model,batchsize):
    batch_size = int(batchsize)
    data_loader = get_dataLoader(batchsize)
    total_num = 0
    i=0
    start = time.time()
    with torch.no_grad():
        for X,Y in data_loader:
            X = X.to("cuda:0")
            torch.cuda.nvtx.range_push("start")
            # torch.cuda.profiler.cudart().cudaProfilerStart()
            model(X)
            torch.cuda.nvtx.range_pop()
            # torch.cuda.profiler.cudart().cudaProfilerStop()
            total_num +=X.shape[0]
            i+=1
            if i==1:
                break
    end = time.time()
    spend = end - start   
    print("inference time : {:.3f} ms Throught : {:.3f}".format(batchsize*1000*spend/total_num,total_num/(spend)))

def inferencePytorchWithNsys(model,batchsize):
    batch_size = int(batchsize)
    data_loader = get_dataLoader(batchsize)
    total_num = 0
    i=0
    start = time.time()
    with torch.no_grad():
        for X,Y in data_loader:
            X = X.to("cuda:0")
            # torch.cuda.nvtx.range_push("start")
            torch.cuda.profiler.cudart().cudaProfilerStart()
            model(X)
            # torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.cudart().cudaProfilerStop()
            total_num +=X.shape[0]
            i+=1
            if i==1:
                break
    end = time.time()
    spend = end - start   
    print("inference time : {:.3f} ms Throught : {:.3f}".format(batchsize*1000*spend/total_num,total_num/(spend)))


if __name__ == "__main__":
    # os.system("docker stop $(docker ps -a -q)  >/dev/null 2>&1 ")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batchsize', type=int, required=True)
    parser.add_argument('--type', type=str, required=True,help='ncu nsys')
    args = parser.parse_args()
    batchsizeList = [args.batchsize]
    model_name = args.model_name
    analysis_type = args.type
    model = creat_model("../models",model_name)
    # batchsizeList = [32]
    # os.system("nvidia-smi -rmc")
    os.system("nvidia-smi -rgc")
    # data_loader = get_dataLoader(batchsize)
    print("NVIDIA TRITION :")
    warmup(model,batchsizeList[0])
    warmup(model,batchsizeList[0])
    print("model: {}".format(model_name))
    for batchsize in batchsizeList:
        # data_loader = load_images_to_batches("./data/imagenet/tiny-imagenet-200/test/images",batchsize)
        if analysis_type == 'ncu':
            inferencePytorchWithNcu(model,batchsize)
        elif analysis_type == 'nsys':
            inferencePytorchWithNsys(model,batchsize)