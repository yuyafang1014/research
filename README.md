# SLIC-Superpixel-Implementation
> The python implementation to make superpixels by slic.

## Requirements:

> gdal 2.x or 3.x

> opencv & opencv_contrib >= 3.1

> skimage 0.14.x or 0.15.x

> matplotlib >= 1.x

## Algorithm description:

> SLIC超像素分割算法分为四部分：
>
> 第一部分是使用GDAL库读取大文件TIF格式的遥感影像（gdal.Open(sys.argv[1])）。
>
> 第二部分是进行slic影像分割。这一部分是利用skimage库自带的slic函数，可以直接进行调用，非常方便。需要注意的是传入的参数格式和个数。
>
> 第三部分是对分割后的结果进行区域合并。这里采用的是基于区域邻接图（RAG）的分层合并算法。该算法通过构造区域邻接图（RAG）并逐步合并颜色相似的区域来达到合并的效果。合并两个相邻区域会生成一个新区域，其中包含来自合并区域的所有像素，直到没有高度相似的区域就结束合并。
>
>第四部分主要是展示实验slic分割和区域合并结果。通过matplotlib提供的pyplot模块来绘制最终的实验结果。

## Run

```python
python slic-superpixel.py input_image.tif 3500 5 20
```

## Results

* original image
![image](https://github.com/yuyafang1014/research/blob/master/input/original_image.png)

* The default value of the compactness is 5
* max_iter is 10
* super_pixel_size is 3500
* sigma is 5
![image](https://github.com/yuyafang1014/research/blob/master/output/Superpixels_--_3500_super_pixel_size.png)

* 20 scale
![image](https://github.com/yuyafang1014/research/blob/master/output/Region_Merging_--_20_scale.png)
