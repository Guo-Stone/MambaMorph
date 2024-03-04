# MambaMorph: a Mamba-based Framework for Medical MR-CT Deformable Registration

![MambaMorph-nocl-v2-github](https://github.com/Guo-Stone/MambaMorph/assets/77957555/d52f5f51-be91-47bc-a11b-cc1a3327aad2)


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
We implement MambaMorph on our reproposed data SR-Reg which is developed from SynthRAD 2023 (https://synthrad2023.grand-challenge.org/). We will upload SR-Reg online after getting necessary permission.

![data-sample](https://github.com/Guo-Stone/MambaMorph/assets/77957555/41cb7576-4fff-49c7-9202-0fc14d7cb9ab)


# Contact
guotao@buaa.edu.cn

# Paper
https://arxiv.org/abs/2401.13934
