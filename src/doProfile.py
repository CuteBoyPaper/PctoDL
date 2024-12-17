import pynvml
import time
def sample_gpu_info(stop_event,handle):
    total_power = 0
    max_power = 0
    max_pcie =0
    total_pcie_tx = 0
    max_pcie_tx = 0
    total_pcie_rx = 0
    max_pcie_rx = 0
    total_gpu_utilization = 0
    max_gpu_utilization = 0
    total_mem_utilization = 0
    max_mem_utilization = 0
    count = 0
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
        max_pcie = max(pcie_rx+pcie_tx,max_pcie)
        # 获取 GPU 利用率
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        total_gpu_utilization+=utilization
        max_gpu_utilization=max(max_gpu_utilization,utilization)
        # 获取内存利用率
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_uti = memory_info.used / memory_info.total * 100
        total_mem_utilization+=mem_uti
        max_mem_utilization=max(max_mem_utilization,mem_uti)
        count = count + 1
        time.sleep(0.01)
    return max_power,total_power/count,max_gpu_utilization,total_gpu_utilization/count,max_mem_utilization,total_mem_utilization/count,max_pcie,(total_pcie_rx+total_pcie_tx)/count
    