# Code for "Mitigating Bias in Set Selection with Noisy Protected Attributes"
This repository contains the code for reproducing the simulations in our paper

**Mitigating Bias in Set Selection with Noisy Protected Attributes**<br>
*Anay Mehrotra and L. Elisa Celis*<br>
Paper: https://arxiv.org/abs/2011.04219


### Running Simulations
- `simulation_a.ipynb` reproduces the simulations from Section 4.2, Section 4.4, and Section 4.5. It also contains an implementation of our algorithm.


- `CIFRank/simulation_b.py` reproduces the simulations from Section 4.3. This code builds (heavily) on the repository [CIFRank](https://github.com/DataResponsibly/CIFRank). To execute this code, run the following command inside the `CIFRank` folder
```
python simulation_b.py "run_exp" "100" "False" "True"
```


### Acknowledgements
- The code for the simulation in Section 4.3 builds upon the repository [CIFRank](https://github.com/DataResponsibly/CIFRank). We include the necessary code in the folder CIFRank.

- `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` are taken from [this](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) tutorial; they are also available in the opencv repository [here](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel).

- `occupations_labels.csv` is taken from the [Occupations dataset](http://bit.ly/2QVfM0K). The folder `Occupations_dataset_images_cropped` contains cropped versions of images from the [Occupations dataset](http://bit.ly/2QVfM0K); we include the code used to crop the images in `simulation_a.ipynb`.

- `census_2010.csv` is taken from US Census Bureau website; it is available [here](https://www.census.gov/topics/population/genealogy/data/2010_surnames.html).


## Citation
```bibtex
  @misc{mehrotra2020mitigating,
        title={Mitigating Bias in Set Selection with Noisy Protected Attributes},
        author={Anay Mehrotra and L. Elisa Celis},
        year={2020},
        eprint={2011.04219},
        archivePrefix={arXiv},
        primaryClass={cs.CY}
  }
```
