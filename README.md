# [TPAMI'25] HAC++
Official Pytorch implementation of **HAC++: Towards 100X Compression of 3D Gaussian Splatting**.
## HAC++ is an enhanced compression method over [HAC](https://github.com/yihangchen-ee/hac/)!

[Yihang Chen](https://yihangchen-ee.github.io), 
[Qianyi Wu](https://qianyiwu.github.io), 
[Weiyao Lin](https://weiyaolin.github.io),
[Mehrtash Harandi](https://sites.google.com/site/mehrtashharandi/),
[Jianfei Cai](http://jianfei-cai.github.io)

[[`Arxiv`](https://arxiv.org/pdf/2501.12255)] [[`Project`](https://yihangchen-ee.github.io/project_hac++/)] [[`Github`](https://github.com/YihangChen-ee/HAC-plus)]

## Links
You are welcomed to check a series of works from our group on 3D radiance field representation compression as listed below:
- 🎉 [CNC](https://github.com/yihangchen-ee/cnc/) [CVPR'24]: efficient NeRF compression! [[`Paper`](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_How_Far_Can_We_Compress_Instant-NGP-Based_NeRF_CVPR_2024_paper.pdf)] [[`Arxiv`](https://arxiv.org/pdf/2406.04101)] [[`Project`](https://yihangchen-ee.github.io/project_cnc/)]
- 🏠 [HAC](https://github.com/yihangchen-ee/hac/) [ECCV'24]: efficient 3DGS compression! [[`Paper`](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01178.pdf)] [[`Arxiv`](https://arxiv.org/pdf/2403.14530)] [[`Project`](https://yihangchen-ee.github.io/project_hac/)]
- 💪 [HAC++](https://github.com/yihangchen-ee/hac-plus/) [ARXIV'25]: an enhanced compression method over HAC! [[`Arxiv`](https://arxiv.org/pdf/2501.12255)] [[`Project`](https://yihangchen-ee.github.io/project_hac++/)]
- 🚀 [FCGS](https://github.com/yihangchen-ee/fcgs/) [ICLR'25]: fast optimization-free 3DGS compression! [[`Paper`](https://openreview.net/pdf?id=DCandSZ2F1)] [[`Arxiv`](https://arxiv.org/pdf/2410.08017)] [[`Project`](https://yihangchen-ee.github.io/project_fcgs/)]
- 🪜 [PCGS](https://github.com/yihangchen-ee/pcgs/) [AAAI'26, Oral]: progressive 3DGS compression! [[`Arxiv`](https://arxiv.org/pdf/2503.08511)] [[`Project`](https://yihangchen-ee.github.io/project_pcgs/)]

## Updates
🔥Jan-2025: HAC++ is now released an enhanced compression method over [HAC](https://github.com/yihangchen-ee/hac/)!

## Overview
<p align="left">
<img src="assets/teaser.png" width=80% height=80% 
class="center">
</p>

HAC++ leverages the relationships between unorganized anchors and a structured hash grid, utilizing their mutual information for context modeling. 
Additionally, HAC++ exploits contextual relationships within anchors to further enhance compression performance. 
To facilitate entropy coding, we utilize Gaussian distributions to precisely estimate the probability of each quantized attribute, 
where an adaptive quantization module is proposed to enable high-precision quantization of these attributes for improved fidelity restoration. 
Moreover, we incorporate an adaptive masking strategy to eliminate invalid Gaussians and anchors.
Overall, HAC++ achieves a remarkable size reduction of over $100\times$ compared to vanilla 3DGS when averaged on all datasets, while simultaneously improving fidelity.

## Performance
<p align="left">
<img src="assets/main_performance.png" width=80% height=80% 
class="center">
</p>


## Installation

We tested our code on a server with Ubuntu 20.04.1, cuda 11.8, gcc 9.4.0.

1. Unzip files
```
cd submodules
unzip diff-gaussian-rasterization.zip
unzip gridencoder.zip
unzip simple-knn.zip
unzip arithmetic.zip
cd ..
```
2. Install environment
```
conda env create --file environment.yml
conda activate HAC_env
```

3. Install ```tmc3``` (for GPCC)

- Please refer to [tmc3 github](https://github.com/MPEGGroup/mpeg-pcc-tmc13) for installation.
- Don't forget to add ```tmc3``` to your environment variable, otherwise you must manually specify its location [in our code](https://github.com/YihangChen-ee/HAC-plus/blob/main/utils/gpcc_utils.py). 
- Tips: ```tmc3``` is commonly located at ```/PATH/TO/mpeg-pcc-tmc13/build/tmc3```.

## Data

First, create a ```data/``` folder inside the project path by 
```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

 - For instance: `./data/blending/drjohnson/`
 - For instance: `./data/bungeenerf/amsterdam/`
 - For instance: `./data/mipnerf360/bicycle/`
 - For instance: `./data/nerf_synthetic/chair/`
 - For instance: `./data/tandt/train/`


### Public Data (We follow suggestions from [Scaffold-GS](https://github.com/city-super/Scaffold-GS))

 - The **BungeeNeRF** dataset is available in [Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing)/[百度网盘[提取码:4whv]](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA). 
 - The **MipNeRF360** scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/). And we test on its entire 9 scenes ```bicycle, bonsai, counter, garden, kitchen, room, stump, flowers, treehill```. 
 - The SfM datasets for **Tanks&Temples** and **Deep Blending** are hosted by 3D-Gaussian-Splatting [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). Download and uncompress them into the ```data/``` folder.

### Custom Data

For custom data, you should process the image sequences with [Colmap](https://colmap.github.io/) to obtain the SfM points and camera poses. Then, place the results into ```data/``` folder.

## Training

To train scenes, we provide the following training scripts: 
 - Tanks&Temples: ```run_shell_tnt.py```
 - MipNeRF360: ```run_shell_mip360.py```
 - BungeeNeRF: ```run_shell_bungee.py```
 - Deep Blending: ```run_shell_db.py```
 - Nerf Synthetic: ```run_shell_blender.py```

 run them with 
 ```
 python run_shell_xxx.py
 ```

The code will automatically run the entire process of: **training, encoding, decoding, testing**.
 - Training log will be recorded in `output.log` of the output directory. Results of **detailed fidelity, detailed size, detailed time** will all be recorded
 - Encoded bitstreams will be stored in `./bitstreams` of the output directory.
 - Evaluated output images will be saved in `./test/ours_30000/renders` of the output directory.
 - Optionally, you can change `lmbda` in these `run_shell_xxx.py` scripts to try variable bitrate.
 - **After training, the original model `point_cloud.ply` is losslessly compressed as `./bitstreams`. You should refer to `./bitstreams` to get the final model size, but not `point_cloud.ply`. You can even delete `point_cloud.ply` if you like :).**


## Contact

- Yihang Chen: yhchen.ee@sjtu.edu.cn

## Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{hac2024,
  title={HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression},
  author={Chen, Yihang and Wu, Qianyi and Lin, Weiyao and Harandi, Mehrtash and Cai, Jianfei},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
```bibtex
@article{hac++2025,
  title={HAC++: Towards 100X Compression of 3D Gaussian Splatting},
  author={Chen, Yihang and Wu, Qianyi and Lin, Weiyao and Harandi, Mehrtash and Cai, Jianfei},
  journal={arXiv preprint arXiv:2501.12255},
  year={2025}
}
```


## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Acknowledgement

 - We thank all authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) for presenting such an excellent work.
 - We thank all authors from [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for presenting such an excellent work. 
 - We thank [Xiangrui](https://liuxiangrui.github.io)'s help on GPCC codec.
