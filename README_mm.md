# MambaMorph: a Mamba-based Backbone with Contrastive Feature Learning for \\Deformable MR-CT Registration  

# Tutorial
Train
```
python ./scripts/torch/train_cross.py --gpu 1 --epochs 1 --batch-size 1 --model-dir output/train_debug --model mm-feat --cl 0.001
```

Test
```
python ./scripts/torch/test_cross.py --gpu 0 --model mm --load-model "/home/guotao/code/voxelmorph-dev/output/train_s46/min_train.pt"
```

# Data:
We implement MambaMorph on our reproposed data SR-Reg. We will post online after getting necessary permission.

# Contact:
guotao@buaa.edu.cn