# Universal Adversarial Perturbations via SGD

This repository contains sample code and interactive Jupyter notebooks for the following work:

* ["Universal Adversarial Perturbations to Understand Robustness of Texture vs. Shape-biased Training"](https://arxiv.org/abs/1911.10364)
* ["Robustness and Transferability of Universal Attacks on Compressed Models"](https://openreview.net/forum?id=HJx08NSnnE) [(AAAI'21 Workshop)](http://federated-learning.org/rseml2021/)

We encourage you to explore these Python notebooks to generate and evaluate your own UAPs. We suggest testing the notebooks for the CIFAR-10 UAPs first.

The notebook **texture-shape.ipynb** visualizes some results discussed in the [paper](https://arxiv.org/abs/1911.10364), exploring the UAPs for texture and shape-biased models.

![slider](uaps_all.png)


The universal attacks on **CIFAR-10** and **ImageNet** models are based on:

* Stochastic gradient descent UAP as proposed by [Shahfahi et al.](https://ojs.aaai.org//index.php/AAAI/article/view/6017)
* Layer maximization attack proposed by [Co et al.](https://arxiv.org/abs/1911.10364)

Universal Adversarial Perturbations (UAPs) generated via Stochastic Gradient Descent (SGD), or referred to as Stochastic Projected Gradient Descent (sPGD) by  [Mummadi et al.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mummadi_Defending_Against_Universal_Perturbations_With_Shared_Adversarial_Training_ICCV_2019_paper.pdf) and [Deng & Karam](https://ieeexplore.ieee.org/document/9191288), has been shown to create more effective UAPs than the originally proposed iterative-DeepFool by [Moosavi-Dezfooli et al.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf) We implement targeted and untargeted versions of the SGD-UAP algorithm.


## ImageNet Dataset
We refer to instructions [here](https://github.com/pytorch/examples/tree/master/imagenet) for downloading and preparing the ImageNet dataset. 


## Acknowledgments
Learn more about the [Resilient Information Systems Security (RISS)](http://rissgroup.org/) group at Imperial College London. Kenneth Co is partially supported by [DataSpartan](http://dataspartan.co.uk/).

Please cite these papers if you use code in this repository as part of a published research project.

```
@article{co2019universal,
  title={Universal Adversarial Perturbations to Understand Robustness of Texture vs. Shape-biased Training},
  author={Co, Kenneth T and Mu{\~n}oz-Gonz{\'a}lez, Luis and Kanthan, Leslie and Glocker, Ben and Lupu, Emil C},
  journal={arXiv preprint arXiv:1911.10364},
  year={2019}
}

@article{matachana2020robustness,
  title={Robustness and Transferability of Universal Attacks on Compressed Models},
  author={Matachana, Alberto G and Co, Kenneth T and Mu{\~n}oz-Gonz{\'a}lez, Luis and Martinez, David and Lupu, Emil C},
  journal={arXiv preprint arXiv:2012.06024},
  year={2020}
}
```
This project is licensed under the MIT License, see the [LICENSE.md](LICENSE.md) file for details.
