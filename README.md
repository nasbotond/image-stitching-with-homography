# ImageStitchingWithHomography

There is a car, and it is equipped with two digital cameras. The cameras are synchronized in time. We have uploaded ten sample images for both cameras. The images from the two cameras are stitched together by a homography that is estimated as: 

(i) The homography is estimated only for one image pair (two images that were taken at the same time). 

(ii) Feature matching by OpenCV or other tools like ASIFT.

(iii) Normalized linear homography estimation.

(iv) Homography estimation is robustified by the standard RANSAC method.
