<h1 align="center"> YouHome Activities-of-Daily-Living (ADL) Dataset </h1>


[![License: Apache](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

The **YouHome ADL dataset** is introduced on this page. The YouHome ADL dataset was gathered in two households that had [YouHome-nodes](#YouHome-node) installed. It includes **20** users' **31** common daily living activities. The dataset has a resolution of 640 x 480 and includes image, illuminance, temperature, humidity, motion, and sound sensor data. There are **isolated daily activities, continuous daily activities sequences, and multiple-user interactive data**. The image data were sliced from videos. The subjects' faces have been blurred for privacy reasons. At the moment, labeled images and sensor data are available.

![](https://github.com/UIUC-ChenLab/YouHome-Dataset/blob/main/readme/house1_2.jpg)


## Table of contents
* [Features](#features)
* [People](#People)
* [Tutorial and Examples](#Tutorial)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Dataset Preparation](#dataset--preparation)
    * [Train](#train)
    * [Test](#test)
* [YouHome-nodes](#YouHome-node)
* [How to Cite?](#citation)
* [Related Repos](#related-repos)





## Features
The YouHome ADL dataset aims to push multiple state-of-the-art smart home tasks forward. The YouHome dataset includes 20 human subjects with and without interaction performing 31 daily activities in two home environments. In all data samples, light conditions are designed to be varied. All human subjects in the dataset are labeled with user ids and bounding box coordinates. The multi-sensor data can feature many tasks, while the below are some representative examples. 

### Indoor-environment Human Detection
Object detection is one of the most famous computer vision tasks; however, there is a lack of dataset focusing on human object detection in the home environment where the light condition can change frequently, and the occlusion of humans can happen suddenly. This differs from outdoor object detection, where the difficulty is majorly on the small size of the object or the similarity between the object and the background color. The YouHome dataset features new challenges in object detection, not only in **light condition changes** or **occluded data** but also in **human objects in different poses** completely different from ordinary standing postures. 
![](https://github.com/UIUC-ChenLab/YouHome-Dataset/blob/main/readme/human_detection_1.jpg)

### Person Re-Identification
Person re-identification is a follow-up task after a human object is detected. The major focus of the person Re-Id is the re-identification of pedestrians in an outdoor environment. Because most of the available datasets are surveillance videos from public places, the posture of pedestrians is mostly limited to walking. In the YouHome dataset, we present a novel concept and challenge for re-ID: the re-identification of indoor users, where users perform different postures than walking on various outfits. This adds more difficulty in recognizing users with color features. When completing the movement in the home environment, users adopt a walking posture and sitting, lying, and other **irregular postures**. In the previous dataset, few provided data on the re-recognition of **various garments**. In addition to including this content, we also included the **body shape and appearance changes** of the same users. 
![](https://github.com/UIUC-ChenLab/YouHome-Dataset/blob/main/readme/Reid_1.jpg)

### Activity Recognition
The YouHome ADL dataset possesses many unique features when performing activity recognition. First, it is a **bias-free** dataset that eliminates the possibility of recognizing activities based solely on their environmental context. Multiple activities are conducted and collected at each location in the YouHome dataset; thus, there is no one-to-one mapping between an activity and a context. 
Existing datasets are collected through either **monitoring or shooting**; the YouHome dataset combines these two techniques and delivers more comprehensive data. In addition, few ADL datasets include **interactive events**, whereas subjects in our dataset not only follow prompts to interact during shooting sessions but also act freely without cues while we simply monitor and collect data. This creates a unique chance to train on clean data while testing on more realistic data. In both households, there are five functional areas: Entrance, Kitchen, Dining Room, Living Room, and Bedroom. Users are expected to perform different activities in each area. In addition to the visual images, **ambient sensor data are also available** for providing additional information. 
Furthermore, the opportunity is provided to identify compound events. In the traditional multi-class classification task, only a single label can be output, whereas our **multi-label compound events** must output an arbitrary number of labels. In contrast to the multi-label classification of object detection, where the feature of different objects is utterly different, the multi-label classification of compound events is difficult due to the similarity of user action postures.
![](https://github.com/UIUC-ChenLab/YouHome-Dataset/blob/main/readme/activity_1.jpg)

### Cross-camera Challenge
Cross camera-view provides a more **heuristic view** of the household environment and enables training and testing on more flexible settings. We provide a sufficient number of camera perspectives so that cross-view challenges can be conducted in a single household or even a single room with various combinations of view angles. Machine learning models can be trained and tested with different combinations of camera views to resemble more realistic scenarios. 
This offers a chance to test the **portability** of the machine learning model with a different partition of training and testing set. With training data in one room and testing data in others, the performance of the model in a new home environment can be tested. In addition, by training the model in one house and testing it in the other, a more practical testing scenario can be achieved. The cross-view challenges can be performed in all tasks mentioned above; the *camera_id* is provided along with other labels.
![](https://github.com/UIUC-ChenLab/YouHome-Dataset/blob/main/readme/cross_1.jpg)

## People
Junhao Pan, Ph.D. Student at University of Illinois at Urbana-Champaign

Zehua(Neo) Yuan, Ph.D. Student at University of Illinois at Urbana-Champaign

Dr. Xiaofan Zhang, Ph.D. from University of Illinois at Urbana-Champaign (Now at Google)

Dr. Deming Chen, Abel Bliss Professor of Engineering at University of Illinois at Urbana-Champaign


## Tutorial and Examples
In this section, we provide training and testing code for activity recognition and human re-identification. During our tutorial, the dataset is split into 8:1:1, Train: Val: Test. The activity recognition is origin from [Pytorch-resnet](https://github.com/kuangliu/pytorch-cifar), the re-identification is origin from [Person-reID-baseline-pytorch](https://github.com/layumi/Person_reID_baseline_pytorch/).

### Prerequisites

- Python >3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 1.10+

### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```

### Dataset & Preparation
Download [YouHome-ADL-Dataset] <font color=red>**++Here need a link**</font>

Preparation: 
For activity recognition, put the images with the same *activity_id* in one folder. 

For human re-identification, put the images with the same *user_id* in one folder.

For the cross-camera challenge, further divide the dataset along with *camera_id*. 

### Train
#### Activity Recognition

Train an activity recognition model by
```
python youhome.py --data_dir {Data_path} --save --save_folder {Save_folder} --epochs {epochs} --batch_size {batch_size} --learning_rate {learning_rate} --decay {decay} --which_gpus {which_gpus} --num_classes {num_classes} --resume {pretrained_model}
```
#### Person Re-identification

`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

Train a model with random erasing by
```bash
python train.py --gpu_ids 0 --name ft_ResNet18 --train_all --batchsize 32  --data_dir your_data_path --erasing_p 0.5
```

### Test
#### Activity Recognition
Test an activity recognition model by
```
python youhome.py --data_dir ${data_dir} --save --save_folder {Save_folder} --batch_size ${batch_size} --which_gpus ${which_gpus} --test 
```

#### Person Re-identification
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--batchsize` batch size.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


##### Evaluation
```bash
python evaluate.py
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.
You may also try `evaluate_gpu.py` to conduct a faster evaluation with GPU.

For mAP calculation, you also can refer to the [C++ code for Oxford Building](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp). We use the triangle mAP calculation.


## YouHome-node
The YouHome-node is a raspberry pi embedded with a pi camera, temperature and humidity sensor, light sensor, microphone, and motion sensor. The pipelined code to use pi-node to collect data is also included in this repository. The pipeline can be run by 
```
bash data_collection.sh {username} {eventname} {eventnumber} {timeinterval}
```
![](https://github.com/UIUC-ChenLab/YouHome-Dataset/blob/main/readme/PiNode.png)

## Citation
Please cite the following paper for the YouHome ADL dataset; it also reports the result of the baseline model.
```bib
@article{youhome2022,
  title={YouHome System and Dataset: Making Your Home Know You Better},
  author={Pan, Junhao and Yuan, Zehua and Zhang, Xiaofan and Chen, Deming},
  journal={IEEE International Symposium on Smart Electronic Systems (IEEE - iSES)},
  year={2022}
}
```



## Related Repos
1. [Person-reID-baseline-pytorch](https://github.com/layumi/Person_reID_baseline_pytorch/)
2. [Yolov5](https://github.com/ultralytics/yolov5)
3. [Pytorch-resnet](https://github.com/kuangliu/pytorch-cifar) 
