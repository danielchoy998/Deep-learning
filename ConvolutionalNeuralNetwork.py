import numpy as np

import numpy as np
 

image = np.random.rand(3, 5, 5)    # RGB 圖像
kernel = np.random.rand(3, 3, 3)   # 3 通道 filter

output = np.zeros((3,3)) # 3 x 3 matrix

def conv2d_with_multi_channel(image, kernel):
    C, H, W = image.shape
    kC, kH, kW = kernel.shape

    assert C == kC, "Channel number must match!"

    out_H = H - kH + 1
    out_W = W - kW + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            # 提取多通道 patch，shape: (C, kH, kW)
            patch = image[:, i:i+kH, j:j+kW]

            # 做 element-wise 相乘 → 再加總所有值
            output[i, j] = np.sum(patch * kernel)

    return output


def conv2d_with_one_channel(image, kernel):
    for i in range(3): 
        for j in range(3):
            region = image[i:i+3, j:j+3]
            output[i, j] = np.sum(region * kernel)
        
    return output

def conv2d_with_padding(image, kernel, padding=1):
    # 原圖尺寸
    H, W = image.shape # 5 x 5
    kH, kW = kernel.shape # 3 x 3

    # 先做 padding
    padded = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

    # 新圖尺寸 = 原圖 + 2*padding
    new_H, new_W = padded.shape

    # 計算輸出尺寸
    out_H = new_H - kH + 1
    out_W = new_W - kW + 1

    # 初始化輸出
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = padded[i:i+kH, j:j+kW]
            output[i, j] = np.sum(region * kernel)

    return output

print(conv2d_with_multi_channel(image, kernel))


