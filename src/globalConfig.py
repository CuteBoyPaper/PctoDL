import threading
import pynvml
from concurrent.futures import ThreadPoolExecutor
class GlobalConfig:
    def __init__(self,platform_data,model_data):
        self.platform = PlatformConfig(platform_data["name"],platform_data["SM Clocks"],platform_data["Memory Clocks"])
        self.model = ModelConfig(model_data["modelName"],model_data["GPU resources"],model_data["data set path"],model_data["Power Cap"]
                                 ,model_data["model repository"],model_data["batch size list"],model_data["log file path"],model_data["max batchsize"]
                                 ,model_data["SLO"],model_data["alpha"],model_data["belta"])
        self.monitor_executor = ThreadPoolExecutor(max_workers=1)
        self.model_executor = ThreadPoolExecutor(max_workers=len(self.model.resources))
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
class PlatformConfig:
    def __init__(self,name,sm_clocks,mem_clocks):
        self.name = name
        sm_clocks.sort()
        mem_clocks.sort()
        self.sm_clocks = sm_clocks
        self.mem_clocks = mem_clocks
class ModelConfig:
    def __init__(self,model_name,resources,data_path,power_cap,model_repository,batch_size_list,log_path,max_bs,slo,alpha,belta):
        self.model_name = model_name
        self.resources = resources
        self.data_path = data_path
        self.power_cap = power_cap
        self.model_repository = model_repository
        self.batch_size_list = batch_size_list
        self.log_path = log_path
        self.max_bs = max_bs
        self.slo = slo
        self.alpha = alpha
        self.belta = belta