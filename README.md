# number_guess
手写数字识别

# 运行方法
训练模型  
python train.py

训练完成后会生成：  
checkpoints/mnist_cnn.pth

预测图片  
python predict.py test.png

输出示例：  
Predicted digit: 7

# 要求
这个脚本用于预测单张图片。你可以传入一张自己的数字图片，例如 test.png。  
图片最好是黑底白字或白底黑字的单个数字  
尺寸不限，程序会自动缩放到 28x28


# kaggle运行
%cd /kaggle/working  
!rm -rf /kaggle/working/number_guess  
!git clone https://github.com/LIKE9426334946/number_guess.git  
%cd /kaggle/working/number_guess  
!python3 train.py
