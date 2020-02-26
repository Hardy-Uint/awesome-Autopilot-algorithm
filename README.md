# 自动驾驶领域算法汇总

> 作者：Tom Hardy
>
> 来源：[3D视觉工坊](https://mp.weixin.qq.com/s?__biz=MzU1MjY4MTA1MQ==&mid=2247484684&idx=1&sn=e812540aee03a4fc54e44d5555ccb843&chksm=fbff2e38cc88a72e180f0f6b0f7b906dd616e7d71fffb9205d529f1238e8ef0f0c5554c27dd7&token=691734513&lang=zh_CN#rd)
>
> 针对自动驾驶领域的传感器标定、感知算法，以及常见仿真工具、国内外自动驾驶公司等进行了汇总~



# 仿真工具

1. [Carla](https://github.com/carla-simulator/carla)
2. [Arisim](https://link.zhihu.com/?target=https%3A//github.com/Microsoft/AirSim)
3. [lgsvl simulator](https://link.zhihu.com/?target=https%3A//www.google.com/search%3Fq%3Dlgsvl%2Bsimulator%26sxsrf%3DACYBGNS8zzXirkwt_eOZccHg0XrmfXOeXg%3A1578662412406%26source%3Dlnms%26tbm%3Disch%26sa%3DX%26ved%3D2ahUKEwje577ij_nmAhWbFIgKHQ5bB-EQ_AUoAXoECAsQAw%26biw%3D1396%26bih%3D657%23imgrc%3DOD-uaRjOazXkrM%3A)
4. [AriSim_Unity](https://github.com/Microsoft/AirSim/tree/master/Unity)
5. [RoadRunner](https://www.rrts.com/)

# 开源框架

1. [Autoware](https://github.com/CPFL/Autoware)
2. [Apollo](https://github.com/ctripcorp/apollo)
3. [ROS](https://www.ros.org/)

# 自动驾驶数据集汇总

1. [Udacity数据](https://link.zhihu.com/?target=https%3A//github.com/udacity/self-driving-car/tree/master/datasets)
2. [Udacity雷达数据](https://github.com/udacity/self-driving-car/tree/master/datasets/CHX)
3. [KITTI数据集](http://www.cvlibs.net/datasets/kitti/index.php)
4. [Mighty AI视觉分类数据](https://mty.ai/dataset/)
5. [Comma AI数据](http://research.comma.ai/)
6. [摄像头数据](https://link.zhihu.com/?target=http%3A//robotcar-dataset.robots.ox.ac.uk/downloads/)
7. [Apollo](https://github.com/ApolloAuto/apollo/tree/master/modules)



# 自动驾驶中的算法汇总

## 综述

1. [A Survey of Autonomous Driving: Common Practices and Emerging Technologies](https://arxiv.org/pdf/1906.05113.pdf)

## 辅助驾驶应用汇总

#### 1、驾驶员状态监控

#### 2、自适应巡航控制（ACC）

#### 3、车道偏离预警（LDW）

#### 4、前方碰撞预警（FCW)

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

## 传感器标定融合

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

#### IMU+摄像头

1. INS-Camera Calibration without Ground Control Points

#### IMU+GPS

#### IMU+GPS+MM

#### 双目+IMU

#### 激光雷达+IMU

## 感知

### 目标检测

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

#### 基于激光雷达点云的3D检测

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

#### 基于摄像头和激光雷达融合的3D目标检测

1. [MLOD: A multi-view 3D object detection based on robust feature fusion method](https://arxiv.org/abs/1909.04163v1)
2. [Multi-Sensor 3D Object Box Refinement for Autonomous Driving](https://arxiv.org/abs/1909.04942?context=cs)
3. [Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving](https://arxiv.org/abs/1906.06310v1)
4. [Improving 3D Object Detection for Pedestrians with Virtual Multi-View Synthesis Orientation Estimation](https://arxiv.org/abs/1907.06777)
5. [Class-specific Anchoring Proposal for 3D Object Recognition in LIDAR and RGB Images](https://arxiv.org/abs/1907.09081)
6. [MVX-Net: Multimodal VoxelNet for 3D Object Detection](https://arxiv.org/pdf/1904.01649.pdf)
7. [Sensor Fusion for Joint 3D Object Detection and Semantic Segmentation](https://arxiv.org/abs/1904.11466v1)
8. [3D Object Detection Using Scale Invariant and Feature Reweighting Networks](https://arxiv.org/abs/1901.02237v1)

### 分割

#### Code

- https://github.com/MarvinTeichmann/MultiNet
- https://github.com/MarvinTeichmann/KittiSeg
- https://github.com/vxy10/p5_VehicleDetection_Unet 
- https://github.com/ndrplz/self-driving-car
- https://github.com/mvirgo/MLND-Capstone
- https://github.com/zhujun98/semantic_segmentation/tree/master/fcn8s_road
- https://github.com/MaybeShewill-CV/lanenet-lane-detection

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
