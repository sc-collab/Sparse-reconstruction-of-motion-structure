# 本项目主要是用于多视角三维重建的稀疏点云重建（只用简单的相机采集图片），后续会增加稠密重建
大体流程为获取图像、解析相机参数、对图片进行图像校正后对其进行稀疏重建
## 安装步骤
### 环境
仓库克隆
```bash
git clone https://github.com/sc-collab/Sparse-reconstruction-of-motion-structure.git
```
创建虚拟环境  
```bash
conda create -n 3D python=3.10
```
安装配置文件
```bash
pip install  -r requirements.txt
```
### 运行必须
Datasets中存有数据集里面应该包含相对应的相机参数k文件！！  
在运行py文件前应检查文件中的路径是否正确！！！
### 图片处理
可以传入视频或图片  
若为视频则用 `frame.py`进行抽帧处理来获取图片后按下述继续操作  
图片则先用`paixu.py`对其进行排序  
### 相机参数获取
后用拍摄图片的相机拍摄imgs中的相同样子的棋盘格图片（要多视角10到20张即可）若没有可自行打印  
将拍好的imgs图片运用`xiangjicanshu.py`获得k.txt  
### 图片矫正
再后来运用k.txt和自己的数据集运用`xiaozhengjibian.py`来对自己的图片进行校正畸变（可以不做这一步但效果不大好）  
### 稀疏重建
最后运用`sfm.py`既可以在jieguo中看到自己的稀疏点云（稀疏点云为少量特征点集合，与稠密点云差距较大）  
点云查看软件可以用MeshLab等  



