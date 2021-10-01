<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="">
   <a href="https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation">
    <img src="images/Logo.jpg" alt="Logo" width="160" height="160">
  </a>

  <h3 align="center">Deep-Active-Learning-Network-for-Brain-MRI-Segmentation</h3>

  <p align="center">
    An active learning solution for brain tumour segmentation using a Bayesian UNet with batch normalisation and max-pool Monte Carlo dropout
    <br />
    <a href="https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation"><strong>Explore the docs »</strong></a>
    <br />
    <br />
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#instructions">Instructions</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

Brain tumour segmentation plays a vital role in computer-aided diagnosis (CAD) by allowing the distinction and identification of various anatomical and physiological features of the disease which can be helpful in localisation of pathology and treatment planning. 
  Deep learning has been trialled for automatic medical image segmentation and can provide a fast and reliable tool for this task. However, these networks require a lot of training data to achieve a high accuracy.
  Active learning is a dynamic form of model learning in which the model can request labels of the samples that it believes are the most informative for learning from an oracle (e.g. a human annotator). This offers a solution to alleviate the annotation burden associated with training a deep learning model. 
  In this work, a framework for AL has been developed specific to brain tumour segmentation. Instead of having a human annotator as an oracle, the oracle is simulated using the labels for the unlabelled dataset since they are available to us.
  Coding references can be seen at the top of the active_learning.py file.
  
### Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)

<!-- GETTING STARTED -->

### Prerequisites

  * Python 3.8.7 64-bit
  ```sh
  https://www.python.org/downloads/release/python-387/y
  ```
  * numpy 1.19.5
  ```sh
  pip install numpy
  ```
 * matplotlib 3.3.3
  ```sh
  pip install matplotlib
  ```
 * nibabel 3.2.1
  ```sh
  pip install nibabel
  ```
  * sklearn 1.0
  ```sh
  pip install scikit-learn
  ```
  * skimage.io
  ```sh
  pip install scikit-image
  ```
  * SimpleITK 2.1.1
  ```sh
  pip install sitk
  ```
  * Hausdorff 0.2.6
  ```sh
  pip install Hausdorff
  ```
  * Pandas 1.2.1
  ```sh
  pip install pandas
  ```
  * tqdm 4.56.0
  ```sh
  pip install tqdm
  ```
 * PyTorch 1.7.1+cu110
  ```sh
  pip install torch
  ```

### Instructions
NOTE: the folders in the OneDrive accompany the files in this Github repo and contain larger folders.

1. Download the original BraTS 2018 data titled 'MICCAI_BraTS_2018_Data_Training' or just the pre-processed data in the 'all_data' folder from https://cityuni-my.sharepoint.com/:f:/g/personal/aaron_mir_city_ac_uk/Eq3XOS6aqotKoW39chEi8RkBU2qcDcFYYOJ-ldfOm9W6dg?e=VW4Uqi
2. If downloading the original BraTS 2018 data, then you must first pre-process the data by running preprocess.py. (NOTE: This takes fairly long)
   ```sh
   https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation/blob/master/preprocess.py
   ```
3. Once you have the preprocessed data, run train_val.py to train the base-segmentation model and save the model weights [GPU is recommended]. 
   Alternatively you can skip this step and use the pre-trained model weights provided on the OneDrive link in the 'models/base_trained' folder.
   The data is already split into train/unlabelled/test but if you wish to create a new data split before running, then please run the data_split.ipynb file.
  ```sh
   https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation/blob/master/data_split.ipynb
   ```
   ```sh
   https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation/blob/master/train_val.py
   ```
4. If you wish to perform a test using the test-set on the base-segmentation model then you can run test.py
   ```sh
   https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation/blob/master/test.py
   ```
5. Configure the parameters in the active_learning.py main() function e.g. random/active learning, number of experiments, number of iterations etc. 
   ```sh
   https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation/blob/master/active_learning.py
   ```
6. Run the active_learning.py file to begin learning iterations and experiments [GPU is recommended].
   ```sh
   https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation/blob/master/active_learning.py
   ```
7. Edit the results.py file to display plots and images of results.
   ```sh
   https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation/blob/master/results.py
   ```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Aaron - Aaron.Mir@city.ac.uk

Project Link: [https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation](https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation)
