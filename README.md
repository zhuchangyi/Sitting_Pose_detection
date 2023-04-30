# Setting_Pose_detection
This project is an implementation of MoveNet which is developed by Google. Inspired by monolesan's fix_posture projected,we are going to set more thresholds and use the deep-learning method.  
## Setting thresholds  
Use the first 30 frame collected by camera to set the average baseline of each keypoint.  
Set the thresholds by comparing the eye position with average value and the tan() value of two eyes is limited, also the lower body is not allowed to show up.  
In order to reduce the error rate of the MoveNet model, we create few counters for multi-frame detection  

## Deep-learning method

# Things on going:  
- [x] Implemente Movenet to generate keypoints  
- [x] Visualization keypoints on real-time by opencv  
- [x] Setting thresholds for classification
- [ ] Dateset collecting(on going)  
- [ ] Build fully connected network training and testing  
The deep learning method we propose is going to lead users to set up their own model which is at the end of the MoveNet network, due to the different device and working environment.
# LICENSES
Copyright 2021 The TensorFlow Authors.  
Changed by Changyi Zhu.

