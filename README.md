# OP3D
![Teaser image](./docs/teaser.png)

Official PyTorch implementation of the ICCV 2023 paper:

**OrthoPlanes: A Novel Representation for Better 3D-Awareness of GANs**<br>
[Honglin He](https://dhlinv.github.io/)<sup>1,2</sup>\*, [Zhuoqian Yang](https://yzhq97.github.io/)<sup>1,3</sup>\*, Shikai Li<sup>1</sup>, [Bo Dai](http://daibo.info/)<sup>1</sup>, [Wayne Wu](https://wywu.github.io/)<sup>1</sup>† <br>
<sup>1</sup>Shanghai AI Laboratory, <sup>2</sup>Tsinghua University, <sup>3</sup>EPFL
<br>\* equal contribution
<br>† corresponding author

[[project page](https://orthoplanes.github.io)] [[paper](https://arxiv.org/abs/2309.15830)]

## Requirements

* We have done all testing and development using V100 and A100 GPUs.
* Use the following commands with Anaconda to create and activate your Python environment:
  - `conda env create -f environment.yml`
  - `conda activate op3d`
* For smpl-pose conditional inference and training:
  - `Pytorch3d` (https://github.com/facebookresearch/pytorch3d) for inverse LBS and mesh-loading.
  - `Kaolin` (https://github.com/NVIDIAGameWorks/kaolin) for computing smpl-mesh SDF.

## Quick Start
Pre-trained networks are stored as *.pkl files that can be referenced using local filenames.

### Model Zoo
| Structure | Dataset | Link | Type |
| --------- |:-------------------:| :-------------------:| :-------------------:|
| Orthoplanes-7-10 | FFHQ | [op3d-ffhq-512-128-withc-7plane.pkl](https://drive.google.com/file/d/1sCWEz-CWYZqT_T30l2qI7Z81jeCRg4iU/view)| Unconditional (w cam-cond) |
| Orthoplanes-8-10 | FFHQ  | [op3d-ffhq-512-128-noc-8plane.pkl](https://drive.google.com/file/d/1tiQ-5Phm9c5gpmnh0Ou-Q251keV3pwvR/view)| Unconditional (w/o cam-cond) |
| Orthoplanes-12-10 | FFHQ | [op3d-ffhq-512-128-withc-12plane.pkl](https://drive.google.com/file/d/1gR-NyfDLZGcPD1lmM_W1ptxatP4f1OSv/view)| Unconditional (w cam-cond) |
| Orthoplanes-8-10 | AFHQv2-Cats | [op3d-afhq-512-128-withc-8plane.pkl](https://drive.google.com/file/d/1xc-Zzh42NAtXdGvbOE1NtWICsKK6m-m5/view)| Unconditional (w cam-cond)|
| Orthoplanes-4-0 | SHHQ | [op3d-shhq-512-64-4plane-linear.pkl](https://drive.google.com/file/d/17wySk8Fh-mIB7FG2dSWqNbjELx7pBycC/view)| Unconditional (w/o cam-cond)|
| Orthoplanes-4-6 | SHHQ | [op3d-shhq-512-64-4plane-frequency.pkl](https://drive.google.com/file/d/1eIHBWjOxa5AP2pcSrX780sGXvybdvT5c/view)| Unconditional (w/o cam-cond)|
| Orthoplanes-8-10 | SHHQ | [op3d-shhq-512-64-8plane-frequency.pkl](https://drive.google.com/file/d/15dlp1r2UvrlIZHNvS02QcBhNbjJf_6Ba/view)| Unconditional (w/o cam-cond)|
| Orthoplanes-8-10 | SHHQ | To be released | Conditional |

- Orthoplanes-$K$-$L$ means that using $K$ planes along each axis with $L$-power positional encoding for each plane.
- Due to the acknowledgement of SHHQ, we will provide only smpl annotations for inference only. For training, we will provide the scripts and test-cases for unit-test only.

### Generating samples
```.bash
# Generate images and shapes (as .mrc files) using pre-trained model

python op3d/gen_samples/unconditional/gen_samples.py --outdir=out \
--trunc=0.5 --shapes=true --seeds=0-3 --network=/path/to/*pkl
```

To visualize a .mrc shape in ChimeraX (https://www.cgl.ucsf.edu/chimerax/):
1. Import the `.mrc` file with `File > Open`
2. Change step size to 1
3. Change level set to 10 for FFHQ, AFHQv2-Cats and 40 for SHHQ.
4. In the `Lighting` menu in the top bar, change lighting to "Full"

## Training and Evaluation
### Preparing datasets
Please refer to EG3D (https://github.com/NVlabs/eg3d) to prepare FFHQ and AFHQv2-Cats dataset. 

For SHHQ, we will provide the preparing scripts before 12.07, 2023.

### Training
```.bash
# Take FFHQ as an example, train from scratch with raw neural rendering resolution=64, using 8 GPUs.
python op3d/train_scripts/train.py --outdir=./save/ffhq --cfg=ffhq --data=FFHQ_512.zip \ 
--batch=32 --gpus=8 --gamma=1 --gen_pose_cond=True

# To save the training time, set metics to none
python op3d/train_scripts/train.py --outdir=./save/ffhq --cfg=ffhq --data=FFHQ_512.zip \ 
--batch=32 --gpus=8 --gamma=1 --gen_pose_cond=True --metrics=none
```

### Evaluation
```.bash
# Take FFHQ as an example, evaluate fid.
python op3d/test_scripts/cal_metrics.py --network=op3d-ffhq-512-128-withc-12plane.pkl \
--gpus=4 --metrics=fid50k_full --data=FFHQ_512.zip
```

## Updates
- [29/11/2023] Code released!
- [28/09/2023] Technical report released!
- [13/07/2023] Our work has been accepted by ICCV2023!

## TODO List
- [x] Release technical report.
- [x] Release code for inference and training of unconditional generation.
- [x] Release pretrained models for unconditional FFHQ, AFHQv2-Cats, SHHQ.
- [ ] Release code for preparing SHHQ dataset.
- [ ] Release code for smpl-pose conditional generation.
- [ ] Release code for NeRF-overfitting Experiments.
- [ ] Release GUI code.

## Citation

```
@inproceedings{he2023orthoplanes,
  title={OrthoPlanes: A Novel Representation for Better 3D-Awareness of GANs},
  author={He, Honglin and Yang, Zhuoqian and Li, Shikai and Dai, Bo and Wu, Wayne},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22996--23007},
  year={2023}
}
```

## Related Works
(ECCV 2022) StyleGAN-Human: A Data-Centric Odyssey of Human Generation 
**[[Demo Video]](https://youtu.be/nIrb9hwsdcI)** | **[[Project Page]](https://stylegan-human.github.io/)** | **[[Paper]](https://arxiv.org/pdf/2204.11823.pdf)**

(ICCV 2023) 3DHumanGAN: 3D-Aware Human Image Generation with 3D Pose Mapping
**[[Project Page]](https://3dhumangan.github.io/)** | **[[Paper]](https://arxiv.org/pdf/2212.07378.pdf)**

## Acknowledgements
This repository uses code from other open-source repositories. We thank the contributors for sharing their wonderful work.
### [Stylegan2-Ada-Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
Karras, Tero, et al. "Training generative adversarial networks with limited data." Advances in neural information processing systems 33 (2020): 12104-12114.
### [EG3D](https://github.com/NVlabs/eg3d)
Chan E R, Lin C Z, Chan M A, et al. Efficient geometry-aware 3D generative adversarial networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 16123-16133.
