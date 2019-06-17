<p align="center"><img width="30%" src="Pet.png" /></p>

--------------------------------------------------------------------------------

# Pet
Pytorch efficient toolbox (Pet) for Computer Vision.

## Introduction
- [x] **Region-base CNN (fast/rpn/faster/mask/keypoint/parsing rcnn, retinanet)**
- [x] **Image Classification**
- [x] **Single Stage Detector (ssd/fssd)**
- [x] **Pose Estimation (body/hand/face simple-baseline)**
- [x] **Face Recognition**
- [ ] **Semantic Segmentation** (in progressing...)
- [ ] **Re-identification** (in progressing...)
- [ ] **Face Keypoints** (in progressing...)

**For different tasks, training/testing scripts and more content under the corresponding path**

### Major features

|                    | norm train | dist train | train by test | cpu train | multi-gpu test | batch test | cpu test |
|--------------------|:----------:|:----------:|:-------------:|:---------:|:--------------:|:----------:|:--------:|
| cls                | ✗          | ✓          | ✓             | ✗         | ✓              | ✓          | ✗        |
| rcnn               | ✗          | ✓          | ✗             | ✗         | ✓              | ✓          | ✗        |
| ssd                | ✗          | ✓          | ✗             | ✗         | ✓              | ✗          | ✗        |
| pose               | ✗          | ✓          | ✗             | ✗         | ✓              | ✓          | ✗        |
| face               | ✗          | ✓          | ✗             | ✗         | ✓              | ✓          | ✗        |
| semseg             | ✗          | ✗          | ✗             | ✗         | ✗              | ✗          | ✗        |

- **Functoins**

  Excel at various tasks of Computer vision.
  
  Provide implementations of latest deep learning algorithms. 
  
  Aim to help developer start own research quickly.
  
- **Features**

  Modularization, flexible configuration.
  
  Implementation of state-of-the-art algorithm in Computer Vison. 
  
  Clear and easy-learning process of training & inference.

- **Contrast（Advance)**

  Support various kinds of tasks in Computer Vison.
  
  Provide numerous high-quality pre-training model. 
  
  Own unique advantages in speed and accuracy.
  
- **Expand**

  Easy to validate new ideas using provided various of basic function.
  
  Code with uniform format and style are easy to expand. 
  
  Update and expand constantly.  Custom extension is supported.

## Installation
 
 Please find detailed installation instructions for PET in [`INSTALL.md`](INSTALL.md).
 
 
## Getting started
 
 Please find detailed tutorial for getting started in [`GETTING_STARTED.md`](docs/GETTING_STARTED.md).


## License

PET is released under the MIT License (refer to the LICENSE file for details).


## Contribute
Feel free to create a pull request if you find any bugs or you want to contribute (e.g., more datasets and more network structures).
