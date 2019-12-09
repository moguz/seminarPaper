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

References
------------
[1] Kanazawa, Angjoo, et al. "Learning category-specific mesh reconstruction from image collections." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
[2] Loper, Matthew, et al. "SMPL: A skinned multi-person linear model." ACM transactions on graphics (TOG) 34.6 (2015): 248.
[3] Anguelov, Dragomir, et al. "SCAPE: shape completion and animation of people." ACM transactions on graphics (TOG). Vol. 24. No. 3. ACM, 2005.
[4] Wah, Catherine, et al. "The caltech-ucsd birds-200-2011 dataset." (2011).
[5] Cashman, Thomas J., and Andrew W. Fitzgibbon. "What shape are dolphins? building 3d morphable models from 2d images." IEEE transactions on pattern analysis and machine intelligence 35.1 (2012): 232-244.
