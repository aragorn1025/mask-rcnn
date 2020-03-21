# Mask R-CNN

## 準備
- 請先安裝 [git][git_url]
- 請先安裝 [Anaconda][anaconda_url]，可依照自己電腦的作業系統進行安裝

[git_url]: https://git-scm.com/downloads/
[anaconda_url]: https://www.anaconda.com/distribution/

## 安裝
請注意CUDA版本
```
conda create --name mask_rcnn python=3.6 -y
conda activate mask_rcnn
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install opencv -c menpo

git clone https://github.com/cocodataset/cocoapi.git
mv cocoapi /opt
cd /opt/cocoapi/PythonAPI
python setup.py build_ext install

pip install -r requirements.txt
```

## 檔案結構
```
./
|- data                           // 訓練所用的資料集
|    |- dataset
|         |- classes              // 類別名稱
|              |- dataset.names
|         |- images
|         |- masks
|- outputs                        // 輸出檔案
|- weights                        // 網路鍵結值
|    |- weights.pth
|- utilities                      // 模組
|    |- datasets                  // 資料集讀取模組
|    |- models                    // 網路模型模組
|    |- tools                     // 其他工具模組
|    |- engine.py                 // 模組主引擎
|- thirdparty                     // 第三方模組
|    |- thirdparty-module-a
|- requirements.txt               // 所需環境
|- detect.py                      // 單張圖片測試
|- train.py                       // 訓練網路
```
