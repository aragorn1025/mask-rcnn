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
pip install -r requirements.txt
```

## 檔案結構
```
./
|- data                    // 資料
|  |- dataset
|- outputs                 // 輸出檔案
|- utilities               // 模組
|  |- datasets             // 資料集讀取模組
|  |- models               // 網路模型模組
|  |- Engine.py            // 模組主引擎
|- weights                 // 網路鍵結值
|  |- weights.pth
|- main.ipynb              // 程式碼執行結果
|- requirements.txt
|- test.py
|- train.py
```

