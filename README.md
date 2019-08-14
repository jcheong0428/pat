# Pose Analysis Toolbox (PAT)

# Intro
This toolbox hosts useful functions to extract features and analyze pose data extracted from OpenPose.  

Ultimately we want this to be a package & class.  
But first let's write functions in separate notebooks in the `notebooks` folders that would be included in the package.

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
