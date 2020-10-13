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
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -y
conda install opencv -c menpo -y
pip install Cython
pip install -r requirements.txt
```

## 檔案結構
```
mask-rcnn
├── src                               // 源程式碼
│   ├── utilities                     // 模組
│   │   ├── datasets                  // 資料集讀取模組
│   │   ├── models                    // 網路模型模組
│   │   ├── tools                     // 其他工具模組
│   │   └── engine.py                 // 模組主引擎
│   ├── thirdparty                    // 第三方模組
│   │   └── torchvision               // PyTorch Vision v0.3.0
│   ├── train.py                      // 訓練網路
│   ├── evaluate.py                   // 評估網路
│   └── detect.py                     // 偵測單張圖片
├── data                              // 資料集
│   └── dataset_name                  // 資料集名稱
│       ├── classes
│       │   ├── dataset_name.class
│       │   └── dataset_name.crowd
│       ├── images
│       │   ├── 000.png
│       │   ├── 001.png
│       │   └── ...
│       └── masks
│           ├── 000.json
│           ├── 001.json
│           └── ...
├── weights                           // 網路鍵結值
│   └── weights.pth
├── outputs                           // 輸出檔案
├── requirements.txt                  // 所需環境
└── README.md                         // 說明文件
```

## 參考
部分程式碼取自 PyTorch torchvision 中，置於``` src/thirdparty ```
