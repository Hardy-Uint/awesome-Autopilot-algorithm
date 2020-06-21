# 自动驾驶领域算法汇总

> 作者：Tom Hardy
>
> 来源：[3D视觉工坊](https://mp.weixin.qq.com/s?__biz=MzU1MjY4MTA1MQ==&mid=2247484684&idx=1&sn=e812540aee03a4fc54e44d5555ccb843&chksm=fbff2e38cc88a72e180f0f6b0f7b906dd616e7d71fffb9205d529f1238e8ef0f0c5554c27dd7&token=691734513&lang=zh_CN#rd)
>
> 针对自动驾驶领域的传感器标定、融合、感知算法，以及常见仿真工具、国内外自动驾驶公司等进行了汇总~



# 仿真工具

2. [英伟达 Drive Constellation](https://www.nvidia.cn/self-driving-cars/drive-constellation/) - 采用逼真的模拟技术，以更安全、更易扩展、 更经济有效的方式推动自动驾驶汽车上路行驶的进程。它利用两台不同服务器的计算能力来提供革命性的云计算平台，从而实现数十亿英里的自动驾驶汽车测试。
3. [优达 self-driving car|开源](https://github.com/udacity/self-driving-car) -用于纳米课程学习
4. [英特尔Carla|开源](https://github.com/carla-simulator/carla) - 用于城市自动驾驶系统的开发、训练和验证的开源模拟器，支持多种传感模式和环境条件的灵活配置
5. [微软 Airsim|开源](https://github.com/Microsoft/AirSim) - 作为人工智能研究的平台，以实验自动驾驶汽车的深度学习，计算机视觉和强化学习算法。
6. [LG LGSVL|开源](https://github.com/lgsvl/simulator) - 帮助开发者集中测试无人驾驶算法，目前平台已经集成了Duckietown, Autoware软件和百度Apollo平台。
7. [百度 Apollo|开源](https://github.com/ApolloAuto/apollo) - 帮助汽车行业及自动驾驶领域的合作伙伴结合车辆和硬件系统，快速搭建一套属于自己的自动驾驶系统。
8. [RoadRunner](https://www.rrts.com/)

# 可视化工具

- [优步ATG AVS|开源](https://avs.auto/#/) - 其主要包括两个repo: [xviz](https://github.com/uber/xviz)处理数据 和 [streetscape.gl](https://github.com/uber/streetscape.gl)进行场景渲染。
- [通用Cruise WorldView|开源](https://github.com/cruise-automation/webviz) -开放可视化组件便于开发者进行无人驾驶数据可视化处理。

# 开源框架

1. [Autoware](https://github.com/CPFL/Autoware)
2. [Apollo](https://github.com/ctripcorp/apollo)
3. [ROS](https://www.ros.org/)

# 自动驾驶数据集汇总

1. [nuScenes](https://www.nuscenes.org/) - 安波福于2019年3月公开了其数据集，并在[GitHub](https://github.com/nutonomy/nuscenes-devkit)公开教程。数据集拥有从波士顿和新加坡收集的1000个“场景”的信息，包含每个城市环境中都有的最复杂的一些驾驶场景。该数据集由140万张图像、39万次激光雷达扫描和140万个3D人工注释边界框组成，是迄今为止公布的最大的多模态3D AV数据集。
2. [H3D - HRI-US](https://usa.honda-ri.com/hdd/introduction/h3d) - 本田研究所于2019年3月发布其无人驾驶方向数据集，相关介绍于[arXiv:1903.01568](https://arxiv.org/abs/1903.01568)介绍。本数据集使用3D LiDAR扫描仪收集的大型全环绕3D多目标检测和跟踪数据集。 其包含160个拥挤且高度互动的交通场景，在27,721帧中共有100万个标记实例。凭借独特的数据集大小，丰富的注释和复杂的场景，H3D聚集在一起，以激发对全环绕3D多目标检测和跟踪的研究。
3. [ApolloCar3D] - 该数据集包含5,277个驾驶图像和超过60K的汽车实例，其中每辆汽车都配备了具有绝对模型尺寸和语义标记关键点的行业级3D CAD模型。该数据集比PASCAL3D +和KITTI（现有技术水平）大20倍以上。
4. [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/raw_data.php) - 数据集为使用各种传感器模式，例如高分辨率彩色和灰度立体相机，Velodyne 3D激光扫描仪和高精度GPS/IMU惯性导航系统，在10-100 Hz下进行6小时拍摄的交通场景。
5. [Cityscape Dataset](https://www.cityscapes-dataset.com/) - 专注于对城市街景的语义理解。 大型数据集，包含从50个不同城市的街景中记录的各种立体视频序列，高质量的像素级注释为5000帧，另外还有一组较大的20000个弱注释帧。 因此，数据集比先前的类似尝试大一个数量级。 可以使用带注释的类的详细信息和注释示例。
6. [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas?pKey=xyW6a0ZmrJtjLw2iJ71Oqg&lat=20&lng=0&z=1.5)-数据集是一个新颖的大规模街道级图像数据集，包含25,000个高分辨率图像，注释为66个对象类别，另有37个类别的特定于实例的标签。通过使用多边形来描绘单个对象，以精细和细粒度的样式执行注释。
7. [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) -剑桥驾驶标签视频数据库（CamVid）是第一个具有对象类语义标签的视频集合，其中包含元数据。 数据库提供基础事实标签，将每个像素与32个语义类之一相关联。 该数据库解决了对实验数据的需求，以定量评估新兴算法。 虽然大多数视频都使用固定位置的闭路电视风格相机拍摄，但我们的数据是从驾驶汽车的角度拍摄的。 驾驶场景增加了观察对象类的数量和异质性。
8. [Caltech数据集](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) - 加州理工学院行人数据集包括大约10小时的640x480 30Hz视频，这些视频来自在城市环境中通过常规交通的车辆。 大约250,000个帧（137个近似分钟的长段）共有350,000个边界框和2300个独特的行人被注释。 注释包括边界框和详细遮挡标签之间的时间对应。 更多信息可以在我们的PAMI 2012和CVPR 2009基准测试文件中找到。
9. [Comma.ai](https://archive.org/details/comma-dataset) - 7.25小时的高速公路驾驶。 包含10个可变大小的视频片段，以20 Hz的频率录制，相机安装在Acura ILX 2016的挡风玻璃上。与视频平行，还记录了一些测量值，如汽车的速度、加速度、转向角、GPS坐标，陀螺仪角度。 这些测量结果转换为均匀的100 Hz时基。
10. [Oxford's Robotic Car](http://robotcar-dataset.robots.ox.ac.uk/) - 超过100次重复对英国牛津的路线进行一年多采集拍摄。 该数据集捕获了许多不同的天气，交通和行人组合，以及建筑和道路工程等长期变化。
11. [伯克利BDD100K数据](https://bdd-data.berkeley.edu/) - 超过100K的视频和各种注释组成，包括图像级别标记，对象边界框，可行驶区域，车道标记和全帧实例分割，该数据集具有地理，环境和天气多样性
12. [Udacity](https://github.com/udacity/self-driving-car/tree/master/datasets) - 为[Udacity Challenges](https://www.udacity.com/self-driving-car)发布的Udacity数据集。 包含ROSBAG训练数据。 （大约80 GB）。
13. [University of Michigan North Campus Long-Term Vision and LIDAR Dataset](http://robots.engin.umich.edu/nclt/) - 包括全方位图像，3D激光雷达，平面激光雷达，GPS和本体感应传感器，用于使用Segway机器人收集的测距。
14. [University of Michigan Ford Campus Vision and Lidar Data Set](http://robots.engin.umich.edu/SoftwareData/Ford) - 基于改进的福特F-250皮卡车的自动地面车辆测试台收集的数据集。 该车配备了专业（Applanix POS LV）和消费者（Xsens MTI-G）惯性测量单元（IMU），Velodyne 3D激光雷达扫描仪，两个推扫式前视Riegl激光雷达和Point Grey Ladybug3全方位摄像头 系统。
15. [DIPLECS Autonomous Driving Datasets (2015)](http://cvssp.org/data/diplecs/) - 通过在Surrey乡村周围驾驶的汽车中放置高清摄像头来记录数据集。 该数据集包含大约30分钟的驾驶时间。 视频为1920x1080，采用H.264编解码器编码。 通过跟踪方向盘上的标记来估计转向。 汽车的速度是从汽车的速度表OCR估算的（但不保证方法的准确性）。
16. [Velodyne SLAM Dataset from Karlsruhe Institute of Technology](http://www.mrt.kit.edu/z/publ/download/velodyneslam/dataset.html) - 在德国卡尔斯鲁厄市使用Velodyne HDL64E-S2扫描仪记录的两个具有挑战性的数据集。
17. [SYNTHetic collection of Imagery and Annotations (SYNTHIA)](http://synthia-dataset.net/) - 包括从虚拟城市渲染的照片般逼真的帧集合，并为13个类别提供精确的像素级语义注释：misc，天空，建筑，道路，人行道，围栏，植被，杆，汽车，标志，行人， 骑自行车的人，车道标记。
18. [CSSAD Dataset](http://aplicaciones.cimat.mx/Personal/jbhayet/ccsad-dataset) - 包括若干真实世界的立体数据集，用于在自动驾驶车辆的感知和导航领域中开发和测试算法。 然而，它们都没有记录在发展中国家，因此它们缺乏在街道和道路上可以找到的特殊特征，如丰富的坑洼，减速器和特殊的行人流。 该立体数据集是从移动的车辆记录的，并且包含高分辨率立体图像，其补充有从IMU，GPS数据和来自汽车计算机的数据获得的定向和加速度数据。
19. [Daimler Urban Segmetation Dataset](http://www.6d-vision.com/scene-labeling) - 包括城市交通中记录的视频序列。 该数据集由5000个经过校正的立体图像对组成，分辨率为1024x440。 500帧（序列的每10帧）带有5个类的像素级语义类注释：地面，建筑，车辆，行人，天空。 提供密集视差图作为参考，但是这些不是手动注释的，而是使用半全局匹配（sgm）计算的。
20. [Self Racing Cars - XSens/Fairchild Dataset](http://data.selfracingcars.com/) - 文件包括来自Fairchild FIS1100 6自由度（DoF）IMU，Fairchild FMT-1030 AHRS，Xsens MTi-3 AHRS和Xsens MTi-G-710 GNSS / INS的测量结果。 事件中的文件都可以在MT Manager软件中读取，该软件可作为MT软件套件的一部分提供，可在此处获得。
21. [MIT AGE Lab](http://lexfridman.com/automated-synchronization-of-driving-data-video-audio-telemetry-accelerometer/) - 由AgeLab收集的1,000多小时多传感器驾驶数据集的一小部分样本。
22. [LaRA](http://www.lara.prd.fr/lara) -巴黎的交通信号灯数据集
23. [KUL Belgium Traffic Sign Dataset](http://www.vision.ee.ethz.ch/~timofter/traffic_signs/) - 具有10000多个交通标志注释的大型数据集，数千个物理上不同的交通标志。 用8个高分辨率摄像头录制的4个视频序列安装在一辆面包车上，总计超过3个小时，带有交通标志注释，摄像机校准和姿势。 大约16000张背景图片。 这些材料通过GeoAutomation在比利时，佛兰德斯地区的城市环境中捕获。
24. [博世小交通灯](https://hci.iwr.uni-heidelberg.de/node/6132) - 用于深度学习的小型交通灯的数据集。
25. [LISA: Laboratory for Intelligent & Safe Automobiles, UC San Diego Datasets](http://cvrr.ucsd.edu/LISA/datasets.html) - 交通标志，车辆检测，交通灯，轨迹模式。
26. [Multisensory Omni-directional Long-term Place Recognition (MOLP) dataset for autonomous driving](http://hcr.mines.edu/code/MOLP.html) - 它是在美国科罗拉多州的一年内使用全向立体相机录制的。[论文](https://arxiv.org/abs/1704.05215)
27. [DeepTesla](https://selfdrivingcars.mit.edu/deeptesla/) - 主要包括tesla在两种不同驾驶模式（human driving和autopilot）下的前置相机录制的视频和车辆的转向控制信号。

# 相关课程

- [[优达学城\] Self-Driving Car Nanodegree Program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) - 教学自动驾驶团队使用的技能和技巧。 可以在[这里](https://medium.com/self-driving-cars/term-1-in-depth-on-udacitys-self-driving-car-curriculum-ffcf46af0c08#.bfgw9uxd9)找到课程大纲 .
- [[多伦多大学\] CSC2541 Visual Perception for Autonomous Driving](http://www.cs.toronto.edu/~urtasun/courses/CSC2541/CSC2541_Winter16.html) - 自动驾驶视觉感知研究生课程。 本课程简要介绍了定位，自我运动估计，自由空间估计，视觉识别（分类，检测，分割）等主题。
- [[INRIA\] Mobile Robots and Autonomous Vehicles](https://www.fun-mooc.fr/courses/inria/41005S02/session02/about?utm_source=mooc-list) - 介绍了对移动机器人和自动驾驶汽车进行编程所需的关键概念。 该课程提供了算法工具，并针对其上周的主题（行为建模和学习），它还将提供Python中的实际示例和编程练习。
- [[格拉斯哥大学\] ENG5017 Autonomous Vehicle Guidance Systems](http://www.gla.ac.uk/coursecatalogue/course/?code=ENG5017) - 介绍自动驾驶仪指导和协调背后的概念，使学生能够设计和实施规划、优化和反应的车辆的指导策略。
- [[David Silver - 优达学城\] How to Land An Autonomous Vehicle Job: Coursework](https://medium.com/self-driving-cars/how-to-land-an-autonomous-vehicle-job-coursework-e7acc2bfe740#.j5b2kwbso) - 来自Udacity的David Silver回顾了他在软件工程背景下从事自动驾驶汽车工作的课程。
- [[斯坦福\] - CS221 Artificial Intelligence: Principles and Techniques](http://stanford.edu/~cpiech/cs221/index.html) - 包含一个简单的自动驾驶项目以及模拟器。
- [[MIT\] 6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/) - 通过构建自动驾驶汽车的应用主题介绍深度学习的实践。

# 相关会议&期刊

1. IEEE ITSC：International Conference on Intelligent Transportation System
2. IEEE ICRA：International Conference on Robotics and Automation
3. IEEE IROS: International Conference on Robotics and System
4. IEEE IV: Intelligent Vehicle Symposium
5. IEEE Robotics and Automation Letters
6. IEEE ICCV: International Conference on Computer Version
7. RSS: Robotics:Science and System
8. IEEE International Conference Vehicular Electronics and Safety
9. CVPR: Conference on Computer Vision and Pattern Recognition
10. NeurIPS: Conference on Neural Information Processing System
11. Towards Autonomous Robotics System
12. ECCV：European Conference on Computer Version
13. AAAI
14. ECC：European Control Conference

# 开源资料总结

[https://github.com/manfreddiaz/awesome-autonomous-vehicles](https://github.com/manfreddiaz/awesome-autonomous-vehicles)

[https://github.com/DeepTecher/awesome-autonomous-vehicle](https://github.com/DeepTecher/awesome-autonomous-vehicle)

[https://github.com/beedotkiran/Lidar_For_AD_references](https://github.com/beedotkiran/Lidar_For_AD_references)

https://github.com/DeepTecher/awesome-autonomous-vehicle

# 自动驾驶中的算法汇总

## 综述

1. [A Survey of Autonomous Driving: Common Practices and Emerging Technologies](https://arxiv.org/pdf/1906.05113.pdf)

## 辅助驾驶应用汇总

#### 1、驾驶员状态监控

#### 2、自适应巡航控制（ACC）

#### 3、车道偏离预警（LDW）

#### 4、前方碰撞预警（FCW)

1. [Forward Vehicle Collision Warning Based on Quick Camera Calibration](http://arxiv.org/abs/1904.12642v1)

#### 5、行人碰撞预警（PCW）

#### 6、智能限速识别（SLI)

#### 7、驾驶员安全带检测

#### 8、自动泊车

#### 9、自动更变车道

#### 10、倒车辅助

#### 11、刹车辅助

#### 12、自动跟车

#### 13、疲劳驾驶检测

#### 14、行驶状态预测

#### 15、停车位检测

1. 霍夫线变换
2. LSD线段检测

## 传感器标定融合

### 相机在线标定

1. A New Technique of Camera Calibration: A Geometric Approach Based on Principal Lines
2. Calibration of fisheye camera using entrance pupil（鱼眼相机）
3. [Forward Vehicle Collision Warning Based on Quick Camera Calibration](http://arxiv.org/abs/1904.12642v1)
4. [Extrinsic camera calibration method and its performance evaluation](http://arxiv.org/abs/1809.11073v1)
5. [A Perceptual Measure for Deep Single Image Camera Calibration](http://arxiv.org/abs/1712.01259v3)

### 传感器标定

1. [RegNet: Multimodal Sensor Registration Using Deep Neural Networks](https://arxiv.org/abs/1707.03167)
2. [CalibNet: Geometrically Supervised Extrinsic Calibration using 3D Spatial Transformer Networks](https://arxiv.org/abs/1803.08181)

### 数据融合

#### 摄像头+Lidar数据融合

1. LiDAR and Camera Calibration using Motion Estimated by Sensor Fusion Odometry
2. Automatic Online Calibration of Cameras and Lasers
3. Automatic Targetless Extrinsic Calibration of a 3D Lidar and Camera by Maximizing Mutual Information
4. Automatic Calibration of Lidar and Camera Images using Normalized Mutual Information
5. Integrating Millimeter Wave Radar with a Monocular Vision Sensor for On-Road Obstacle Detection Applications
6. [Automatic extrinsic calibration between a camera and a 3D Lidar using 3D point and plane correspondences](http://arxiv.org/abs/1904.12433v1)
7. [A Novel Method for the Extrinsic Calibration of a 2D Laser Rangefinder and a Camera](http://arxiv.org/abs/1603.04132v4)
8. [LiDAR and Camera Calibration using Motion Estimated by Sensor Fusion Odometry](http://arxiv.org/abs/1804.05178v1)
9. [Reflectance Intensity Assisted Automatic and Accurate Extrinsic Calibration of 3D LiDAR and Panoramic Camera Using a Printed Chessboard](http://arxiv.org/abs/1708.05514v1)

#### IMU+单目摄像头

1. INS-Camera Calibration without Ground Control Points
2. [HKUST-Aerial-Robotics/VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) - SOTA，IROS 2018,IMU和（单目）摄像头融合的校正方法，用来校准IMU和相机之间的时间偏移。

#### IMU+GPS

#### IMU+GPS+MM

#### 双目+IMU

#### 激光雷达+IMU

## 感知

### 目标检测

#### 目标检测常见的trick

1. Focal loss
2. OHEM
3. S-OHEM
4. GHM
5. Soft NMS
6. Multi Scale Training/Testing
7. Mix up（数据增强）
8. Label Smoothing
9. Warm Up
10. Box Refinement/Voting（预测框微调/投票法）
11. RoIAlign RoI对齐
12. KL-Loss
13. large batch BN
14. loss synchronization
15. automatic BN fusion

#### 2D目标检测

| Detector                 | VOC07 (mAP@IoU=0.5) | VOC12 (mAP@IoU=0.5) | COCO (mAP@IoU=0.5:0.95) | Published In |
| ------------------------ | ------------------- | ------------------- | ----------------------- | ------------ |
| R-CNN                    | 58.5                | -                   | -                       | CVPR'14      |
| SPP-Net                  | 59.2                | -                   | -                       | ECCV'14      |
| MR-CNN                   | 78.2 (07+12)        | 73.9 (07+12)        | -                       | ICCV'15      |
| Fast R-CNN               | 70.0 (07+12)        | 68.4 (07++12)       | 19.7                    | ICCV'15      |
| Faster R-CNN             | 73.2 (07+12)        | 70.4 (07++12)       | 21.9                    | NIPS'15      |
| YOLO v1                  | 66.4 (07+12)        | 57.9 (07++12)       | -                       | CVPR'16      |
| G-CNN                    | 66.8                | 66.4 (07+12)        | -                       | CVPR'16      |
| AZNet                    | 70.4                | -                   | 22.3                    | CVPR'16      |
| ION                      | 80.1                | 77.9                | 33.1                    | CVPR'16      |
| HyperNet                 | 76.3 (07+12)        | 71.4 (07++12)       | -                       | CVPR'16      |
| OHEM                     | 78.9 (07+12)        | 76.3 (07++12)       | 22.4                    | CVPR'16      |
| MPN                      | -                   | -                   | 33.2                    | BMVC'16      |
| SSD                      | 76.8 (07+12)        | 74.9 (07++12)       | 31.2                    | ECCV'16      |
| GBDNet                   | 77.2 (07+12)        | -                   | 27.0                    | ECCV'16      |
| CPF                      | 76.4 (07+12)        | 72.6 (07++12)       | -                       | ECCV'16      |
| R-FCN                    | 79.5 (07+12)        | 77.6 (07++12)       | 29.9                    | NIPS'16      |
| DeepID-Net               | 69.0                | -                   | -                       | PAMI'16      |
| NoC                      | 71.6 (07+12)        | 68.8 (07+12)        | 27.2                    | TPAMI'16     |
| DSSD                     | 81.5 (07+12)        | 80.0 (07++12)       | 33.2                    | arXiv'17     |
| TDM                      | -                   | -                   | 37.3                    | CVPR'17      |
| FPN                      | -                   | -                   | 36.2                    | CVPR'17      |
| YOLO v2                  | 78.6 (07+12)        | 73.4 (07++12)       | -                       | CVPR'17      |
| RON                      | 77.6 (07+12)        | 75.4 (07++12)       | 27.4                    | CVPR'17      |
| DeNet                    | 77.1 (07+12)        | 73.9 (07++12)       | 33.8                    | ICCV'17      |
| CoupleNet                | 82.7 (07+12)        | 80.4 (07++12)       | 34.4                    | ICCV'17      |
| RetinaNet                | -                   | -                   | 39.1                    | ICCV'17      |
| DSOD                     | 77.7 (07+12)        | 76.3 (07++12)       | -                       | ICCV'17      |
| SMN                      | 70.0                | -                   | -                       | ICCV'17      |
| Light-Head R-CNN         | -                   | -                   | 41.5                    | arXiv'17     |
| YOLO v3                  | -                   | -                   | 33.0                    | arXiv'18     |
| SIN                      | 76.0 (07+12)        | 73.1 (07++12)       | 23.2                    | CVPR'18      |
| STDN                     | 80.9 (07+12)        | -                   | -                       | CVPR'18      |
| RefineDet                | 83.8 (07+12)        | 83.5 (07++12)       | 41.8                    | CVPR'18      |
| SNIP                     | -                   | -                   | 45.7                    | CVPR'18      |
| Relation-Network         | -                   | -                   | 32.5                    | CVPR'18      |
| Cascade R-CNN            | -                   | -                   | 42.8                    | CVPR'18      |
| MLKP                     | 80.6 (07+12)        | 77.2 (07++12)       | 28.6                    | CVPR'18      |
| Fitness-NMS              | -                   | -                   | 41.8                    | CVPR'18      |
| RFBNet                   | 82.2 (07+12)        | -                   | -                       | ECCV'18      |
| CornerNet                | -                   | -                   | 42.1                    | ECCV'18      |
| CenterNet                |                     |                     | 45.1                    |              |
| HSD                      |                     |                     |                         |              |
| PFPNet                   | 84.1 (07+12)        | 83.7 (07++12)       | 39.4                    | ECCV'18      |
| Pelee                    | 70.9 (07+12)        | -                   | -                       | NIPS'18      |
| HKRM                     | 78.8 (07+12)        | -                   | 37.8                    | NIPS'18      |
| M2Det                    | -                   | -                   | 44.2                    | AAAI'19      |
| R-DAD                    | 81.2 (07++12)       | 82.0 (07++12)       | 43.1                    | AAAI'19      |
| ScratchDet               | 84.1 (07++12)       | 83.6 (07++12)       | 39.1                    | CVPR'19      |
| Libra R-CNN              | -                   | -                   | 43.0                    | CVPR'19      |
| Reasoning-RCNN           | 82.5 (07++12)       | -                   | 43.2                    | CVPR'19      |
| FSAF                     | -                   | -                   | 44.6                    | CVPR'19      |
| AmoebaNet + NAS-FPN      | -                   | -                   | 47.0                    | CVPR'19      |
| Cascade-RetinaNet        | -                   | -                   | 41.1                    | CVPR'19      |
| TridentNet               | -                   | -                   | 48.4                    | ICCV'19      |
| DAFS                     | **85.3 (07+12)**    | 83.1 (07++12)       | 40.5                    | ICCV'19      |
| Auto-FPN                 | 81.8 (07++12)       | -                   | 40.5                    | ICCV'19      |
| FCOS                     | -                   | -                   | 44.7                    | ICCV'19      |
| FreeAnchor               | -                   | -                   | 44.8                    | NeurIPS'19   |
| DetNAS                   | 81.5 (07++12)       | -                   | 42.0                    | NeurIPS'19   |
| NATS                     | -                   | -                   | 42.0                    | NeurIPS'19   |
| AmoebaNet + NAS-FPN + AA | -                   | -                   | 50.7                    | arXiv'19     |
| EfficientDet             | -                   | -                   | **51.0**                | arXiv'19     |
| SaccadeNet               |                     |                     | 40.4                    | CVPR2020     |

#### 基于单目图像的3D检测

1. [Task-Aware Monocular Depth Estimation for 3D Object Detection](https://arxiv.org/abs/1909.07701)
2. [M3D-RPN: Monocular 3D Region Proposal Network for Object Detection](https://arxiv.org/abs/1907.06038v1)
3. [Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud](https://arxiv.org/pdf/1903.09847.pdf)
4. [Monocular 3D Object Detection and Box Fitting Trained End-to-End Using Intersection-over-Union Loss](https://arxiv.org/pdf/1906.08070.pdf)
5. [Disentangling Monocular 3D Object Detection](https://arxiv.org/pdf/1905.12365v1.pdf)
6. [Shift R-CNN: Deep Monocular 3D Object Detection with Closed-Form Geometric Constraints](https://arxiv.org/pdf/1905.09970.pdf)
7. [Monocular 3D Object Detection via Geometric Reasoning on Keypoints](https://arxiv.org/abs/1905.05618?context=cs.CV)
8. [Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction](https://arxiv.org/abs/1904.01690)
9. [GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving](https://arxiv.org/abs/1903.10955)
10. [Accurate Monocular Object Detection via Color-Embedded 3D Reconstruction for Autonomous Driving](https://arxiv.org/abs/1903.11444?context=cs.CV)
11. [Task-Aware Monocular Depth Estimation for 3D Object Detection](https://arxiv.org/abs/1909.07701)
12. [M3D-RPN: Monocular 3D Region Proposal Network for Object Detection](https://arxiv.org/abs/1907.06038v1)
13. [YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud](https://arxiv.org/abs/1808.02350)
14. [YOLO4D: A ST Approach for RT Multi-object Detection and Classification from LiDAR Point Clouds]()
15. [Deconvolutional Networks for Point-Cloud Vehicle Detection and Tracking in Driving Scenarios](https://arxiv.org/abs/1808.07935)
16. [PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/abs/1812.04244)
17. [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/abs/1803.06199)
18. [FVNet: 3D Front-View Proposal Generation for Real-Time Object Detection from Point Clouds](https://arxiv.org/abs/1903.10750v1)
19. [Part-A^2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud](https://arxiv.org/abs/1907.03670v1)

#### 基于立体视觉的3D检测

1. [Object-Centric Stereo Matching for 3D Object Detection](https://arxiv.org/pdf/1909.07566.pdf)
2. [Triangulation Learning Network: from Monocular to Stereo 3D Object Detection](https://arxiv.org/pdf/1906.01193.pdf)
3. [Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving](http://www.cs.cornell.edu/~yanwang/project/plidar/)
4. [Stereo R-CNN based 3D Object Detection for Autonomous Driving](https://arxiv.org/pdf/1902.09738.pdf)
5. [IDA-3D: Instance-Depth-Aware 3D Object Detection from Stereo Vision for Autonomous Driving（CVPR2020）](http://openaccess.thecvf.com/content_CVPR_2020/papers/Peng_IDA-3D_Instance-Depth-Aware_3D_Object_Detection_From_Stereo_Vision_for_Autonomous_CVPR_2020_paper.pdf) [源代码](https://github.com/swords123/IDA-3D)
6. [Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation（CVPR2020)](https://arxiv.org/abs/2004.03572) [源代码](https://github.com/zju3dv/disprcn)
7. [DSGN: Deep Stereo Geometry Network for 3D Object Detection(CVPR2020)](https://arxiv.org/abs/2001.03398) [源代码](https://github.com/chenyilun95/DSGN)

#### 基于激光雷达点云/Voxel的3D检测

1. [End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds]()
2. [Vehicle Detection from 3D Lidar Using Fully Convolutional Network(百度早期工作)](https://arxiv.org/abs/1608.07916)
3. [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/pdf/1711.06396.pdf)
4. [Object Detection and Classification in Occupancy Grid Maps using Deep Convolutional Networks](https://arxiv.org/pdf/1805.08689.pdf)
5. [RT3D: Real-Time 3-D Vehicle Detection in LiDAR Point Cloud for Autonomous Driving](https://www.onacademic.com/detail/journal_1000040467923610_4dfe.html)
6. [BirdNet: a 3D Object Detection Framework from LiDAR information](https://arxiv.org/pdf/1805.01195.pdf)
7. [LMNet: Real-time Multiclass Object Detection on CPU using 3D LiDAR](https://arxiv.org/pdf/1805.04902.pdf)
8. [HDNET: Exploit HD Maps for 3D Object Detection](https://link.zhihu.com/?target=http%3A//proceedings.mlr.press/v87/yang18b/yang18b.pdf)
9. [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)
10. [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
11. [IPOD: Intensive Point-based Object Detector for Point Cloud](https://arxiv.org/abs/1812.05276v1)
12. [PIXOR: Real-time 3D Object Detection from Point Clouds](http://www.cs.toronto.edu/~wenjie/papers/cvpr18/pixor.pdf)
13. [DepthCN: Vehicle Detection Using 3D-LIDAR and ConvNet](https://www.baidu.com/link?url=EaE2zYjHkWvF33nsET2eNvbFGFu8-D3wWPia04uyKm95jMetHsSv3Zk-tODPGm5clsgCUgtVULsZ6IQqv0EYS_Z8El7Zzh57XzlJroSkaOuC8yv7r1XXL4bUrM2tWrTgjwqzfMV2tMTnFNbMOmHLTkUobgMg7HKoS6WW6PfQzkG&wd=&eqid=8f320cfa0005b878000000055e528b6d)
14. [YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud](https://arxiv.org/abs/1808.02350)
15. [Voxel-FPN: multi-scale voxel feature aggregation in 3D object detection from point clouds](https://arxiv.org/ftp/arxiv/papers/1907/1907.05286.pdf)
16. [STD: Sparse-to-Dense 3D Object Detector for Point Cloud](https://arxiv.org/abs/1907.10471)
17. [Fast Point R-CNN](https://arxiv.org/abs/1908.02990)
18. [StarNet: Targeted Computation for Object Detection in Point Clouds](https://arxiv.org/abs/1908.11069)
19. [Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection](https://arxiv.org/abs/1908.09492v1)
20. [LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving](https://arxiv.org/abs/1903.08701v1)
21. [Structure Aware Single-stage 3D Object Detection from Point Cloud（CVPR2020)](http://openaccess.thecvf.com/content_CVPR_2020/html/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.html) [源代码](https://github.com/skyhehe123/SA-SSD)
22. [MLCVNet: Multi-Level Context VoteNet for 3D Object Detection（CVPR2020)](https://arxiv.org/abs/2004.05679) [源代码](https://github.com/NUAAXQ/MLCVNet)
23. [3DSSD: Point-based 3D Single Stage Object Detector（CVPR2020）](https://arxiv.org/abs/2002.10187) [源代码](https://github.com/tomztyang/3DSSD)
24. [LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention（CVPR2020）](https://arxiv.org/abs/2004.01389) [源代码](https://github.com/yinjunbo/3DVID)
25. [PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection(CVPR2020)](https://arxiv.org/abs/1912.13192) [源代码](https://github.com/sshaoshuai/PV-RCNN)
26. [Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud（CVPR2020）](https://arxiv.org/abs/2003.01251) [源代码](https://github.com/WeijingShi/Point-GNN)

#### 基于摄像头和激光雷达融合的3D目标检测

1. [MLOD: A multi-view 3D object detection based on robust feature fusion method](https://arxiv.org/abs/1909.04163v1)
2. [Multi-Sensor 3D Object Box Refinement for Autonomous Driving](https://arxiv.org/abs/1909.04942?context=cs)
3. [Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving](https://arxiv.org/abs/1906.06310v1)
4. [Improving 3D Object Detection for Pedestrians with Virtual Multi-View Synthesis Orientation Estimation](https://arxiv.org/abs/1907.06777)
5. [Class-specific Anchoring Proposal for 3D Object Recognition in LIDAR and RGB Images](https://arxiv.org/abs/1907.09081)
6. [MVX-Net: Multimodal VoxelNet for 3D Object Detection](https://arxiv.org/pdf/1904.01649.pdf)
7. [Sensor Fusion for Joint 3D Object Detection and Semantic Segmentation](https://arxiv.org/abs/1904.11466v1)
8. [3D Object Detection Using Scale Invariant and Feature Reweighting Networks](https://arxiv.org/abs/1901.02237v1)
9. [End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection（CVPR2020）](https://arxiv.org/abs/2004.03080) [源代码](https://github.com/mileyan/pseudo-LiDAR_e2e)

#### 车道线检测

1. https://github.com/cardwing/Codes-for-Lane-Detection

### 分割

#### Code

- https://github.com/MarvinTeichmann/MultiNet
- https://github.com/MarvinTeichmann/KittiSeg
- https://github.com/vxy10/p5_VehicleDetection_Unet 
- https://github.com/ndrplz/self-driving-car
- https://github.com/mvirgo/MLND-Capstone
- https://github.com/zhujun98/semantic_segmentation/tree/master/fcn8s_road
- https://github.com/MaybeShewill-CV/lanenet-lane-detection

#### 分割常见的trick

> 常用的ASPP、空洞卷积、多分辨率融合(MRF)、ENet中的轻量化思想、deeplab系列中的CRF都是。

#### 语义分割

1. DeconvNet [https://arxiv.org/pdf/1505.04366.pdf] [2015]
2. U-Net [https://arxiv.org/pdf/1505.04597.pdf] 
3. SegNet [https://arxiv.org/pdf/1511.00561.pdf] [2016]
4. FCN [https://arxiv.org/pdf/1605.06211.pdf] [2016]
5. ENet [https://arxiv.org/pdf/1606.02147.pdf] [2016]
6. DilatedNet [https://arxiv.org/pdf/1511.07122.pdf] [2016]
7. PixelNet [https://arxiv.org/pdf/1609.06694.pdf] [2016]
8. RefineNet [https://arxiv.org/pdf/1611.06612.pdf] [2016]
9. FRRN [https://arxiv.org/pdf/1611.08323.pdf] [2016]
10. LRR [https://arxiv.org/pdf/1605.02264.pdf] [2016]
11. MultiNet [https://arxiv.org/pdf/1612.07695.pdf] [2016]
12. DeepLab [https://arxiv.org/pdf/1606.00915.pdf] [2017]
13. LinkNet [https://arxiv.org/pdf/1707.03718.pdf] [2017]
14. DenseNet [https://arxiv.org/pdf/1611.09326.pdf] [2017]
15. ICNet [https://arxiv.org/pdf/1704.08545.pdf] [2017]
16. ERFNet [http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf] 
17. PSPNet [https://arxiv.org/pdf/1612.01105.pdf,https://hszhao.github.io/projects/pspnet/] [2017]
18. GCN [https://arxiv.org/pdf/1703.02719.pdf] [2017]
19. DUC, HDC [https://arxiv.org/pdf/1702.08502.pdf] [2017]
20. Segaware [https://arxiv.org/pdf/1708.04607.pdf] [2017]
21. PixelDCN [https://arxiv.org/pdf/1705.06820.pdf] [2017]
22. FPN [http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf] [2017]
23. ShuffleSeg [https://arxiv.org/pdf/1803.03816.pdf] [2018]
24. AdaptSegNet [https://arxiv.org/pdf/1802.10349.pdf] [2018]
25. TuSimple-DUC [https://arxiv.org/pdf/1702.08502.pdf] [2018]
26. R2U-Net [https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf] [2018]
27. Attention U-Net [https://arxiv.org/pdf/1804.03999.pdf] [2018]
28. DANet [https://arxiv.org/pdf/1809.02983.pdf] [2018]
29. ShelfNet [https://arxiv.org/pdf/1811.11254.pdf] [2018]
30. LadderNet [https://arxiv.org/pdf/1810.07810.pdf] [2018]
31. ESPNet [https://arxiv.org/pdf/1803.06815.pdf] [2018]
32. DFN [https://arxiv.org/pdf/1804.09337.pdf] [2018]
33. EncNet [https://arxiv.org/pdf/1803.08904.pdf] [2018]
34. DenseASPP [http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf] [2018]
35. Unet++ [https://arxiv.org/pdf/1807.10165.pdf] [2018]
36. Fast-SCNN [https://arxiv.org/pdf/1902.04502.pdf] [2019]
37. HRNet [https://arxiv.org/pdf/1904.04514.pdf] [2019]
38. PSANet [https://hszhao.github.io/papers/eccv18_psanet.pdf] [2018]
39. UPSNet [https://arxiv.org/pdf/1901.03784.pdf] [2019]
40. DFANet [https://arxiv.org/pdf/1904.02216.pdf] [2019]
41. ExtremeC3Net [https://arxiv.org/pdf/1908.03093.pdf] [2019]

#### 实例分割

1. FCIS [https://arxiv.org/pdf/1611.07709.pdf]
2. MNC [https://arxiv.org/pdf/1512.04412.pdf]
3. DeepMask [https://arxiv.org/pdf/1506.06204.pdf]
4. SharpMask [https://arxiv.org/pdf/1603.08695.pdf]
5. Mask-RCNN [https://arxiv.org/pdf/1703.06870.pdf]
6. RIS [https://arxiv.org/pdf/1511.08250.pdf]
7. FastMask [https://arxiv.org/pdf/1612.08843.pdf]
8. PANet [https://arxiv.org/pdf/1803.01534.pdf] [2018]
9. TernausNetV2 [https://arxiv.org/pdf/1806.00844.pdf] [2018]
10. MS R-CNN [https://arxiv.org/pdf/1903.00241.pdf] [2019]
11. AdaptIS [https://arxiv.org/pdf/1909.07829.pdf] [2019]
12. Pose2Seg [https://arxiv.org/pdf/1803.10683.pdf] [2019]
13. YOLACT [https://arxiv.org/pdf/1904.02689.pdf] [2019]
14. PloarMask[https://arxiv.org/abs/1909.13226]



### 目标跟踪

1. [Fully-Convolutional Siamese Networks for Object Tracking]()
2. [Learning Multi-Domain Convolutional Neural Networks for Visual Tracking]()
3. [Multi-object Tracking with Neural Gating Using Bilinear LSTM]()
4. [Simple Online and Realtime Tracking]()
5. [Simple Online and Realtime Tracking with a Deep Association Metric]()
6. [Online Object Tracking: A Benchmark]()
7. [Visual Tracking: An Experimental Survey]()
8. [Multi-target Tracking]()
9. [Multiple Object Tracking: A Literature Review]()
10. [Survey on Leveraging Deep Neural Networks for Object Tracking]()
11. [Tracking by Learning Discriminative Saliency Map with Convolutional Neural Network](http://arxiv.org/pdf/1502.06796)
12. [DeepTrack: Learning Discriminative Feature Representations by Convolutional Neural Networks for Visual Tracking](http://www.bmva.org/bmvc/2014/files/paper028.pdf)
13. [Learning a Deep Compact Image Representation for Visual Tracking](http://winsty.net/papers/dlt.pdf)
14. [Hierarchical Convolutional Features for Visual Tracking,ICCV 2015](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ma_Hierarchical_Convolutional_Features_ICCV_2015_paper.pdf)
15. [Visual Tracking with fully Convolutional Networks, ICCV 2015](https://github.com/scott89/FCNT) 
16. [Learning Multi-Domain Convolutional Neural Networks for Visual Tracking](https://github.com/HyeonseobNam/MDNet)

### 深度图补全&修复

1、[HMS-Net: Hierarchical Multi-scale Sparsity-invariant Network for Sparse Depth Completion](https://arxiv.org/abs/1808.08685)

2、[Sparse and noisy LiDAR completion with RGB guidance and uncertainty](https://arxiv.org/abs/1902.05356)

3、[3D LiDAR and Stereo Fusion using Stereo Matching Network with Conditional Cost Volume Normalization](https://arxiv.org/pdf/1904.02917.pdf)

4、[Deep RGB-D Canonical Correlation Analysis For Sparse Depth Completion](https://arxiv.org/pdf/1906.08967.pdf)

5、[Confidence Propagation through CNNs for Guided Sparse Depth Regression](https://arxiv.org/abs/1811.01791)

6、[Learning Guided Convolutional Network for Depth Completion](https://arxiv.org/pdf/1908.01238.pdf)

7、[DFineNet: Ego-Motion Estimation and Depth Refinement from Sparse, Noisy Depth Input with RGB Guidance](http://arxiv.org/abs/1903.06397)

8、[PLIN: A Network for Pseudo-LiDAR Point Cloud Interpolation](https://arxiv.org/abs/1909.07137)

9、Depth Completion from Sparse LiDAR Data with Depth-Normal Constraints

### 车道线检测

1. Key Points Estimation and Point Instance Segmentation Approach for Lane Detection
2. Multi-lane Detection Using Instance Segmentation and Attentive Voting
3. A Learning Approach Towards Detection and Tracking of Lane Markings
4. Real time Detection of Lane Markers in Urban Streets
5. Real-Time Stereo Vision-Based Lane Detection System
6. An Empirical Evaluation of Deep Learning on Highway Driving
7. Real-Time Lane Estimation using Deep Features and Extra Trees Regression
8. Accurate and Robust Lane Detection based on Dual-View Convolutional Neutral Network
9. DeepLanes: E2E Lane Position Estimation using Deep NNs
10. Deep Neural Network for Structural Prediction and Lane Detection in Traffic Scene
11. End-to-End Ego Lane Estimation based on Sequential Transfer Learning for Self-Driving Cars
12. Deep Learning Lane Marker Segmentation From Automatically Generated Labels
13. VPGNet: Vanishing Point Guided Network for Lane and Road Marking Detection and Recognition
14. Spatial as Deep: Spatial CNN for Traffic Scene Understanding
15. Towards End-to-End Lane Detection: an Instance Segmentation Approach
16. LaneNet: Real-Time Lane Detection Networks for Autonomous Driving
17. 3D-LaneNet: E2E 3D multiple lane detection
18. End-to-end Lane Detection through Differentiable Least-Squares Fitting
19. Robust Lane Detection from Continuous Driving Scenes Using Deep Neural Networks
20. Efficient Road Lane Marking Detection with Deep Learning
21. LineNet: a Zoomable CNN for Crowdsourced High Definition Maps Modeling in Urban Environments
22. Lane Marking Quality Assessment for Autonomous Driving



### 交通灯和信号检测

1. Vision for Looking at Traffic Lights: Issues, Survey, and Perspectives
2. Vision-Based Traffic Sign Detection and Analysis for Intelligent Driver Assistance Systems: Perspectives and Survey
3. Self-Driving Cars: A Survey
4. Traffic-Sign Detection and Classification in the Wild
5. A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification





# 自动驾驶公司汇总

| 公司名称 |          主要业务          |        base         |
| :------: | :------------------------: | :-----------------: |
|  Waymo   |          自动驾驶          |        美国         |
| Velodyne |     激光雷达、无人驾驶     |      美国硅谷       |
|   Uber   |          自动驾驶          |      美国硅谷       |
| Mobileye |      高级驾驶辅助系统      | 以色列（Intel收购） |
|   博世   |     智能汽车、辅助驾驶     |  德国（斯图加特）   |
|   苹果   |          自动驾驶          |      美国加州       |
|   谷歌   |          自动驾驶          |      美国加州       |
|  特斯拉  |     智能电车、自动驾驶     |      美国硅谷       |
|  英伟达  | 智能芯片、自动驾驶解决方案 |      上海/加州      |
| Minieye  |          自动驾驶          | 深圳/南京/北京/苏州 |
|  AutoX   |          自动驾驶          |      美国硅谷       |
|  Voyage  |          自动驾驶          |      美国硅谷       |
| MaxiEye  |          智能驾驶          |        上海         |
| 图森未来 |        智能驾驶卡车        |        北京         |
| 西井科技 |        智能驾驶卡车        |        上海         |
| 驭势科技 |          自动驾驶          |   上海/北京/深圳    |
| 纵目科技 |       ADAS、辅助驾驶       |        上海         |
|   百度   |         智能驾驶部         |      北京/深圳      |
| 文远知行 |          智能驾驶          |   广州/上海/北京    |
|   阿里   |    智能物流车、智能驾驶    |   杭州/上海/北京    |
|   腾讯   |      自动驾驶解决方案      |      北京/深圳      |
| 仙途智能 |         智能环卫车         |        上海         |
| 滴滴出行 |          自动驾驶          |        北京         |
|   吉利   |          智能驾驶          |      上海/宁波      |
| 小鹏汽车 |     智能电车、辅助驾驶     | 广州/上海/北京/硅谷 |
| 蔚来汽车 |          智能电车          |        上海         |
|   华为   |      自动驾驶解决方案      | 杭州/深圳/上海/西安 |
|  地平线  |   自动驾驶芯片及解决方案   |   南京/北京/上海    |
| 四维图新 |        车联网、地图        |        北京         |
| Monmenta |          自动驾驶          | 苏州/北京/斯图加特  |
|  寒武纪  |        智能驾驶芯片        |        北京         |
| 禾多科技 |        自动驾驶方案        |      北京/上海      |
| 飞步科技 |     无人驾驶、辅助驾驶     |        杭州         |
| 智加科技 |        无人驾驶卡车        |   苏州/上海/北京    |
| 奇点汽车 |    智能汽车系统、车联网    |      北京/上海      |
| 经纬恒润 | 智能驾驶、车联网、汽车电子 |      北京/上海      |
| 径卫视觉 |          辅助驾驶          |        上海         |
| 深兰科技 |  自动驾驶公交、智能环卫车  |        上海         |
| 魔视智能 |   自动驾驶和高级辅助驾驶   |        上海         |
| 欧菲智能 |       ADAS、辅助驾驶       |        上海         |
|   上汽   |     辅助驾驶、自动驾驶     |        上海         |
| 威马汽车 |          智能汽车          |        上海         |
| 海康威视 | 辅助驾驶、自动驾驶解决方案 |      杭州/上海      |
| Nullmax  |      自动驾驶解决方案      |        上海         |
