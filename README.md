<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
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
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Medical image segmentation plays a vital role in computer-aided diagnosis (CAD) by allowing the distinction and identification of various anatomical and physiological features of disease which can be helpful in localisation of pathology and treatment planning. 
       Use of deep learning for automatic medical image segmentation has been promoted as a solution to human error when performing medical image segmentation. Over time, the performance of these automatic medical segmentation algorithms has increased significantly with some outperforming humans (Haenssle et al., 2018). This is due to networks with deep architectures and the increase in available annotated datasets for training such models. Nevertheless, producing annotations for these datasets, specifically medical images, demands an expert and involves intense, time-consuming labour. Compared with natural images, medical images are usually grayscale in colour and have low-contrast which means that some regions in the image may have similar intensity and texture to neighbouring regions. Additionally, while these datasets are available, they are still not available in significant amounts for some areas of interest. Furthermore, most fully supervised models for automatic medical image segmentation have variable performance. Therefore, there is usually a need to refine the results from an automatic segmentation. In most real life medical applications, a human-assisted medical image segmentation program is used (Olabarriaga and Smeulders, 2001). With deep learning, the next logical step would become a human-in-the-loop segmentation program which integrates user input to refine segmentation predictions. This appears to be the preferred approach due to the high-stakes nature of medical image segmentation and the black-box nature of deep learning models. A human-in-the-loop strategy combines human and machine intelligence in order to maximise accuracy. Active learning (AL) is a cornerstone of the human-in-the-loop strategy and is the process of deciding which data to sample for human annotation during the loop (Budd et al., 2021). At every iteration of the AL loop, the model is expected to become more accurate.
       This work contains the proposal for a deep interactive, AL refinement network for medical image segmentation. The dataset used will be the BraTS (Brain Tumour Segmentation) 2016 and 2017 datasets from the medical segmentation decathlon website (Simpson et al., 2019). The BraTS datasets holds 484 pixel-labelled brain MRI volumes in the training set and 337 MRI volumes in the testing set. The class labels for the test set are unavailable hence the training set will have to be split into training and testing sets. The MRI volumes themselves are each of size 240 x 240 x 155 pixels and include the T1-weighted, T2-weighted, T1-Gd and Fluid-Attenuated Inversion Recovery (FLAIR) MRI modalities. The tumours are labelled according to their tumour region type e.g. edema, non-enhancing and enhancing tumour. Necrotic tumour tissue is not labelled in this dataset.

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

### Installation
INSERT INSTRUCTIONS
1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```JS
   const API_KEY = 'ENTER YOUR API';
   ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Aaron - [Email](Aaron.Mir@city.ac.uk)

Project Link: [https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation](https://github.com/Assassinsarms/Deep-Active-Learning-Network-for-Medical-Image-Segmentation)
