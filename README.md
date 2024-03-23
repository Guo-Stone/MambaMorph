# MambaMorph: a Mamba-based Framework for Medical MR-CT Deformable Registration

# Tutorial
Install Mamba via https://github.com/state-spaces/mamba

### Framework

![MambaMorph-nocl-v2-github](E:\Science Research\Medical Image\Paper\IMG\MambaMorph-nocl-v2-github.png)

### Result

![Result](E:\Science Research\Medical Image\Paper\IMG\Result.png)

### **Train**

```
python ./scripts/torch/train_cross.py --gpu 1 --epochs 1 --batch-size 1 --model-dir output/train_debug --model mm-feat
```

### **Test**

```
python ./scripts/torch/test_cross.py --gpu 0 --model mm-feat --load-model "/home/guotao/code/voxelmorph-dev/output/train_s46/min_train.pt"
```

# Data
We implement MambaMorph on our brain MR-CT data SR-Reg which is developed from SynthRAD 2023 (https://synthrad2023.grand-challenge.org/). SR-Reg is provided under a CC-BY-NC 4.0 International license and available at https://drive.google.com/drive/folders/1qxUM-PuvWe1S6GvWudyKUXY8p_jnP_gN?usp=drive_link. When you use SR-Reg, please cite MambaMorph.



### Data sample

![data-sample](E:\Science Research\Medical Image\Paper\IMG\data-sample.png)

# Contact
guotao@buaa.edu.cn

# Paper
https://arxiv.org/abs/2401.13934

# Citation

@article{guo2024mambamorph,  

  title={MambaMorph: a Mamba-based Framework for Medical MR-CT Deformable Registration},  

  author={Guo, Tao and Wang, Yinuo and Shu, Shihao and Chen, Diansheng and Tang, Zhouping and Meng, Cai and Bai, Xiangzhi},  

  journal={arXiv preprint arXiv:2401.13934},  

  year={2024}  

}