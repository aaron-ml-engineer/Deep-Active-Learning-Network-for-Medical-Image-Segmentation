<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="">
   <a href="https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation">
    <img src="images/Logo.jpg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Deep-Active-Learning-Network-for-Brain-MRI-Segmentation</h3>

  <p align="center">
    An active learning solution using a UNet with batch normalisation and max-pool Monte Carlo dropout
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
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

Medical image segmentation plays a vital role in computer-aided diagnosis (CAD) by allowing the distinction and identification of various anatomical and physiological features of disease which can be helpful in localisation of pathology and treatment planning. 


### Built With

* [Python](https://www.python.org/)


<!-- GETTING STARTED -->
## Getting Started

INSERT INSTRUCTIONS

### Prerequisites

  * numpy
  ```sh
  pip install numpy
  ```
 * matplotlib
  ```sh
  pip install matplotlib
  ```
 * nibabel
  ```sh
  pip install nibabel
  ```
 * SimpleITK
  ```sh
  pip install sitk
  ```
 * PyTorch
  ```sh
  pip install torch
  ```

### Instructions

1. Download the BRATS 2018 data from [INSERT ONEDRIVE LINK]
2. Run train_val.py to train the base-segmentation model and save the weights.
3. Configure the parameters in the active_learning.py main() function e.g. number of experiments, number of iterations etc.
   ```sh
   https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation/blob/master/train_val.py
   ```
4. Run the active_learning.py file to begin active learning iterations and experiments.
   ```sh
   https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation/blob/master/active_learning.py
   ```

<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Aaron - Aaron.Mir@city.ac.uk

Project Link: [https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation](https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation)
