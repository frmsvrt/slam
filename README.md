*toy SLAM for fun
---
![](misc/promo.png)
What we have:
 - feature extraction using **Oriented FAST and Rotated BRIEF**
 - feature detection/tracking with **cv2.goodFeaturesToTrack**
 - Find **F**, **Rotation** and **Transform** matricies using **RANSAC**
 - camera pose estimation and points triangulation
 - visualization with **pangolin viwer**
 - graph (constraints?) optimization with **g2o** optimizer
