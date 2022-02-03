import os

for k in range(1,100):
  os.system('python3 train_cifar10.py')
  cmmd = 'cp checkpoint/ckpt.pth models/'+'%d_ckpt.pth'%(k+1)
  os.system(cmmd)

