# Pose Analysis Toolbox (PAT)
<div align="center">
<img src="assets/pat.jpg"
     alt="Pose Analysis Toolbox"
     style="align: center; width: 150px;" />
</div>

-----------------

# Overview
A hackathon project initiated at [MIND 2019](http://mindsummerschool.org/) at Dartmouth College.

High level idea: Create useful functions that preprocess, analyze, and visualize [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) data. Can be used to predict and generate taxonomy of social and non-social poses and interactions.  

# Presentation
Link to presentation goes here.

# Installation.
1. Clone the repository.
2. Install in development mode
```
pip install -e .
```
3. Check out the example notebook `notebooks/Examples.ipynb`
4. For more data, download the Sherlock OpenPose data and extract to `notebooks/output/json`.

# Features
- [x] Load data
- [x] Plot and inspect data
- [x] Extract just the pose_2d keypoints
- [x] Filter based on personid
- [x] Extract Distance Matrix across keypoints
- [ ] Normalize keypoints to common space. Scale, rotate?.
- [ ] Extract Entropy.
- [ ] Extract Synchrony.
- [ ] Extract Center of Mass per frame.
- [ ] Example of comparing distance between 2 or multiple people.
- [ ] Extract Velocity across frame.
- [ ] Extract Acceleration across frame.

# Reference
https://github.com/CMU-Perceptual-Computing-Lab/openpose  
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
