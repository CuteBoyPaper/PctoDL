import os
import sys
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import time
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

def inference(model):
    model_name = model["model_name"]
    result=model["data"]
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
    inference_statistics = client.get_inference_statistics(model_name)
    end_total_time,end_total_count = get_success_inference_stats(inference_statistics)
    lantecy = (end_total_time-start_total_time)/(end_total_count-start_total_count)    
    return lantecy,batchsize*1000/lantecy


def stop_all_docker():
    """
    before test or after test
    clear all active docker container
    """
    return os.system("docker stop $(docker ps -a -q)  >/dev/null 2>&1 ")


def config_gpu(gpu_resource):
    """
    config gpu resource before each triton starting
    """
    os.system("sudo nvidia-cuda-mps-control -d " + "> /dev/null 2>&1 ")
    server_id = os.popen("echo get_server_list | nvidia-cuda-mps-control").readlines()
    # logger.info(server_id)
    if len(server_id) == 0:
        stop_all_docker()
        server_id = os.popen("echo get_server_list | nvidia-cuda-mps-control").readlines()[0].strip('\n')
    else:
        server_id = server_id[0].strip('\n')

    gpu_set_cmd = "echo set_active_thread_percentage {} {} | sudo nvidia-cuda-mps-control".format(server_id,
                                                                                                gpu_resource)
    gpu_resource_set = os.popen(gpu_set_cmd).readlines()[0].strip('\n')

    if float(gpu_resource_set) != gpu_resource:
        print("Failed to config gpu resource")
        return False
    else:
        return True
        print("Success to set gpu resource: {}".format(float(gpu_resource_set)))
    return True



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
    cmd_create_triton_server += "nvcr.io/nvidia/tritonserver:24.10-py3 tritonserver "
    # cmd_create_triton_server += "nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver "
    # cmd_create_triton_server += "--model-control-mode=explicit "
    cmd_create_triton_server += "--model-repository=/models "
    cmd_create_triton_server += "> /dev/null 2>&1 "
    os.system(cmd_create_triton_server)
    print(cmd_create_triton_server)

    print("Waiting for TRITON Server to be ready at http://localhost:{}...".format(port1))
    live = "http://localhost:{}/v2/health/live".format(port1)
    ready = "http://localhost:{}/v2/health/ready".format(port1)
    count = 0
    while True:
        live_command = os.popen(
            "curl -i -m 1 -L -s -o /dev/null -w %{http_code} " + live).readlines()[0]
        ready_command = os.popen(
            "curl -i -m 1 -L -s -o /dev/null -w %{http_code} " + ready).readlines()[0]
        if live_command == "200" and ready_command == "200":
            print("TRITON server is ready now. ")
            break
        # sleep for 1 s
        time.sleep(5)
        count += 1
        if count > 30:
            return False
    return True