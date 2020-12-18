# Codebase for a Highly Versatile Facial Expression Recognition System
*Team FILO*

Wei Cui, Kewei Qiu and Chenhao Gong

## Required Software and Packages

This system was completed using Python 3 (ver. 3.7). Please consult Python's version documentations for compability details. 

To run the system on your local device, the installation of the following Python packages is required: 

- NumPy
- Sklearn
- Tensorflow (`Keras`)
- OpenCV (`cv2`)
- Glob
- Matplotlab

## To Run this System

- To perform the evaluation of our system on FER2013 database, you need to download it into your local device. 
  Since the FER2013 is too large to upload in our repo, you can find the download link here: https://www.kaggle.com/deadskull7/fer2013.
- Change the direction to this repository in your local device;
- Run `python3 main.py` directly, this will run compare_models program which constructs and evaluate accuracy of different models;
- Follow the prompts. Note that the responses are case-sensitive.
- Put one of your own photo into the root directory of the project
- Comment the second last line and uncomment the last line in main.py
- Change the file name of the function input to the photo you just put into the root directory
- Rerun `python3 main.py`, this will run recognize_emotion program which detect faces in the photo and recognize emotions. Results are shown in standard out
- Please note that: face detection in recognize_emotion program is set to auto mode by default, if the result is unsatisfiable, please change the second function input of recognize_emotion from 'auto' to 'manual', then follow the prompts.
- Please also note that: recognize_emotion program has low accuracy and needs further improvement
