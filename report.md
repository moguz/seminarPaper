Seminar Report
=======

Introduction
------------

Suppose that you are shown a picture of a bird, which you have not seen before in your life. Your brain immediately understands that the 
object in the picture is a bird, therefore all the previous knowledge you have gathered through past experiences regarding <em>birds</em> 
in general are attached to the bird image that you have just seen. Only by seeing a single image, you can guess the size of the bird, 
how would the bird looks like from other angles and whether the feathers on the back of the bird look similar to those you see on the image.

This paper is tackling the same problem, which is learning a predictor which predicts 3D shape, camera pose and the mesh texture at the same time, given a single image in inference time. The training data only consists of annotated 2D image collection.

<img src="bird_image.png" width="40%"><img src="bird_3d.png" width="40%">

The problem of 3D reconstruction from a single image is ill-posed<sup>[1]</sup>. If we had multiple images of the same object, then we could expolot multiple-view geometry, such as formulating the problem as a convex variatonal method. However, in this paper, we are only able to reconstruct 3D shape because the predictor already knows the mean shape of a bird in inference time.

Related Work
----------------
Prior works tackled inferring 3D shape problem with different perspectives. Some methods tried to learn deformable models using 3D ground truth data.<sup>[2][3]</sup> The drawback of these methods are that obtaining 3D ground truth data is hard and/or expensive. Especially in cases like using animals as objects, like in CUB-200-2011 dataset<sup>[4]</sup>, 2D annotated image collection is easier to obtain than 3D scans.

<img src="relatedwork_1.png" width="20%"><img src="relatedwork_2.png" width="12%">

Second group of related works use only 2D annotated image collection as training data, similar to this paper. However, these methods require annotations also in test time. For instance, [5] aims to infer 3D shapes of dolphins using 2D images. That method uses keypoints correspondences and segmentation masks in the training time and segmentation masks during test time. The reason behind requiring annotations during test time is that these methods use fitting-based inference, therefore they minimize some kind of loss function during inference time. Our paper, however, is a prediction-based inference model, which directly predicts the 3D shape given the input image.

<img src="relatedwork_3.png" width="30%">

Third group of related works employ different 3D shape representations than deformable meshes. There are a number of possible representations for 3D shape, such as voxels, point cloud or octree. These methods generally require a stronger supervision, such as 3D ground truth data or images from multiple-views. Moreover, the choice of deformable meshes come with several advantages; associating semantic keypoints with mesh vertices and inferring mesh texture as an RGB image in a canonical appearance space.<sup>[1]</sup>

<img src="relatedwork_4.png" width="30%">Representing 3D shape with point cloud<sup>[6]</sup>

To summarize, this paper differs from related work primarily in 3 aspects:
* Deformable model representation and directly inferring 3D shape
* Learning from only 2D image collection
* Ability to infer texture

Methodology
------------
<img src="methodology.png" width="100%">
The aim of the paper is to learn a predictor which is capable of inferring the full 3D representation of an object, given a single 2D unannotated image. The full 3D representation consists of 3D shape, which is parametrized as deformable mesh, camera pose and texture. In order to achieve this, a 2-stage architecture has been used. In the first stage, the input image is fed to an encoder module. The encoder is a convolutional neural network (CNN) whose structure is ResNet-18 model followed by a convolutional layer and two fully-connected layers. Encoder module takes the input image and represents it in a shared latent space of size 200, which is then used by the modules in the second stage.


<img src="encoder.png" width="100%">
In the second stage of the architecture, the latent representation is shared across 3 prediction modules, namely shape prediction, camera pose prediction and texture prediction. The details of those prediction modules and design decisions will be discussed below, however it is important to know the internal architecture of these modules. Shape prediction and camera prediction modules are just linear layers and texture flow module consists of 5 upconvolution layers.

The main dataset used in this paper is CUB-200-2011, which has 6000 training and and test images if 200 species of birds.<sup>[1][4]</sup> Every image in the dataset is annotated with bounding box, visibility indicator, locations of semantic keypoints (can be imagined as the tail, the head etc.) and segmentation masks. The authors filtered out 5% of the data whose semantic keypoints are mostly non-visible.

**Shape Prediction**
Arguably, the most important prediction module of the paper is shape prediction. The reason is that 3D shape is the most informative component in full 3D representation. 


References
------------
[1] Kanazawa, Angjoo, et al. "Learning category-specific mesh reconstruction from image collections." Proceedings of the European Conference on Computer Vision (ECCV). 2018.  
[2] Loper, Matthew, et al. "SMPL: A skinned multi-person linear model." ACM transactions on graphics (TOG) 34.6 (2015): 248.  
[3] Anguelov, Dragomir, et al. "SCAPE: shape completion and animation of people." ACM transactions on graphics (TOG). Vol. 24. No. 3. ACM, 2005.  
[4] Wah, Catherine, et al. "The caltech-ucsd birds-200-2011 dataset." (2011).  
[5] Cashman, Thomas J., and Andrew W. Fitzgibbon. "What shape are dolphins? building 3d morphable models from 2d images." IEEE transactions on pattern analysis and machine intelligence 35.1 (2012): 232-244.  
[6] Fan, Haoqiang, Hao Su, and Leonidas J. Guibas. "A point set generation network for 3d object reconstruction from a single image." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.  
