# Face-Pose-Net
This page contains DCNN model and python code to robustly estimate 6 degrees of freedom, 3D face pose from an unconstrained image, without the use of face landmark detectors. The method is described in the paper:

F. Chang, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "FacePoseNet: Making a Case for Landmark-Free Face Alignment", in arXiv, 2017 [1].

This release bundles up our FacePoseNet (FPN) with the Face Renderer from Masi et al. [2,5], which is available separately from this project page.

The result is an end-to-end pipeline that seamlessly estimates facial pose and produces multiple rendered views to be used for face alignment and data augmentation.
