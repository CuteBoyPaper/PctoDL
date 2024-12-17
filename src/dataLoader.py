import os
import numpy as np
from PIL import Image
def load_images_to_batches(image_dir, batch_size):
    """
    从指定文件夹中读取图像，并按指定的 batch_size 组织成 [batchsize, 3, 224, 224] 的数组。
    仅返回完整的批次。

    :param image_dir: str, 图片文件夹路径
    :param batch_size: int, 每个批次的大小
    :return: list, 包含每个完整批次的 NumPy 数组列表
    """
    batches = []


    image_files = [file for file in os.listdir(image_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    

    current_batch = []

    for i in range(len(image_files)):
        if len(batches)==10:
            break
        if i==len(image_files)-1:
            i = 0
        image_file = image_files[i]

        image_path = os.path.join(image_dir, image_file)
        try:
            
            with Image.open(image_path) as img:
                
                img = img.resize((224, 224))
                
                
                img_array = np.array(img, dtype=np.float32)
                
            
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)

                
                if img_array.shape[2] != 3:
                    raise ValueError(f"Image {image_file} is not in RGB format.")

                
                img_array = img_array.transpose((2, 0, 1))
                
            
                current_batch.append(img_array)

            
                if len(current_batch) == batch_size:
                    batches.append(np.stack(current_batch))
                    current_batch = []

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    
    
    #只保留10个batch，每个调度周期只有10个batch
    return batches