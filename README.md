# MambaMorph: a Mamba-based Framework for Medical MR-CT Deformable Registration


![MambaMorph-nocl-v2](https://github.com/Guo-Stone/MambaMorph/assets/77957555/a3310649-2e46-4842-bf1e-44c6110db12b)


# Tutorial
Install Mamba via https://github.com/state-spaces/mamba

Train
```
python ./scripts/torch/train_cross.py --gpu 1 --epochs 1 --batch-size 1 --model-dir output/train_debug --model mm-feat --cl 0.001
```

Test
```
python ./scripts/torch/test_cross.py --gpu 0 --model mm --load-model "/home/guotao/code/voxelmorph-dev/output/train_s46/min_train.pt"
```

# Data
We implement MambaMorph on our reproposed data SR-Reg. We will post online after getting necessary permission.

# Contact
guotao@buaa.edu.cn

# Paper
https://arxiv.org/abs/2401.13934
