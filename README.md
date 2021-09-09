<!-- ABOUT THE PROJECT -->
## Age and Emotion Recognition

This project's purpose is to create a multi-task model for age and emotion recognition. This model combines the ideas of 3 papers:
* [End-to-End Multi-Task Learning with Attention](https://arxiv.org/pdf/1803.10704.pdf)
* [Context-Aware Emotion Recognition](https://arxiv.org/pdf/1908.05913.pdf)
* [Facial expression and attributes recognition based on multi-task learning of lightweight neural networks](https://arxiv.org/pdf/2103.17107.pdf)


## Model details
The model consists of 3 main modules: face and context modules (as proposed in [Context-Aware Emotion Recognition](https://arxiv.org/pdf/1908.05913.pdf)) and a newly added gating module.

### Face module

The face module receives as an input an aligned and 112*112 face images. Based on the [lightweight neural networks paper](https://arxiv.org/pdf/2103.17107.pdf), we utilized a backbone pretrained with face recognition dataset before finetuning with age and emotion datasets. In this case, the backbone we use for this module is resnet100 trained with MegaFace dataset by [Insightface](https://github.com/deepinsight/insightface). Moreover, the attention modules are added like the proposed method by the paper [End-to-End Multi-Task Learning with Attention](https://github.com/lorenmt/mtan).

### Context module
The face module receives as an input a 224*224 images with faces being cropped out of the image. Based on the [Context-Aware Emotion Recognition](https://arxiv.org/pdf/1908.05913.pdf). The model we choose to use is ImageNet pretrained ResNest 50 proposed from the paper [ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf). Moreover, the attention modules are added like the proposed method by the paper [End-to-End Multi-Task Learning with Attention](https://github.com/lorenmt/mtan).

### Gating module
The Gating module receives as an input a 224*224 whole images with faces. The model we choose to use is ResNest 50 [ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf) pretrained with ImageNet.


## Loss and Optimizers
We use Cross Entropy Loss for each task. Besides, we adopt techniques as proposed in [Loss-Balanced Task Weighting to Reduce Negative Transfer in Multi-Task Learning](https://ojs.aaai.org//index.php/AAAI/article/view/5125) or [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115.pdf) to reduce the effects of negative transfer.


For optimizers, asides from normal Adam optimizer configuation, we utilize [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf) to deal with negative transfer or [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412.pdf) to make the model more robust and less sensitive to noises in the data.

## Training
All values and arguments for training are given and modified within the config.json file. Please run the following command to start training

```
python3 train.py -c config.json
```

## Note
This project is simply a combination of the ideas presented above. This project is not created specifically to any datasets. Therefore, to apply this model, please modify your dataloader and customize the trainer.

## Reference
* End-to-End Multi-Task Learning with Attention: [Paper](https://arxiv.org/pdf/1803.10704.pdf) [Code](https://github.com/lorenmt/mtan)
* Context-Aware Emotion Recognition: [Paper](https://arxiv.org/pdf/1908.05913.pdf)
* Facial expression and attributes recognition based on multi-task learning of lightweight neural networks: [Paper](https://arxiv.org/pdf/2103.17107.pdf) [Code](https://github.com/HSE-asavchenko/face-emotion-recognition)
* Loss-Balanced Task Weighting to Reduce Negative Transfer in Multi-Task Learning: [Paper](https://ojs.aaai.org//index.php/AAAI/article/view/5125)
* Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics: [Paper](https://arxiv.org/pdf/1705.07115.pdf)
* Insightface: [Code](https://github.com/deepinsight/insightface)
* ResNeSt: Split-Attention Networks: [Paper](https://arxiv.org/pdf/2004.08955.pdf) [Code](https://github.com/zhanghang1989/ResNeSt.git)
* Gradient Surgery for Multi-Task Learning: [Paper](https://arxiv.org/pdf/2001.06782.pdf) [Code](https://github.com/WeiChengTseng/Pytorch-PCGrad.git)
* Sharpness-Aware Minimization for Efficiently Improving Generalization: [Paper](https://arxiv.org/pdf/2010.01412.pdf) [Code](https://github.com/davda54/sam.git)
* Pytorch Template: [Code](https://github.com/victoresque/pytorch-template.git)


