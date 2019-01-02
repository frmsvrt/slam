*toy SLAM for fun
___
![](misc/promo.png)
What we have:
 - feature extraction using **Oriented FAST and Rotated BRIEF**
 - feature detection/tracking with **cv2.goodFeaturesToTrack**
 - Find **F**, **Rotation** and **Transform** matricies using **RANSAC**
 - camera pose estimation and points triangulation
 - visualization with **pangolin viwer**
 - graph (constraints?) optimization with **g2o** optimizer
---
Dependencies:
- numpy
- cv2
- pangolin
- g2o
---
Usage:
`d3d=1 d2d=1 python slam.py kitty.mp4`

where `d3d` is `3D map viewer` and `d2d` is `video viewer`
