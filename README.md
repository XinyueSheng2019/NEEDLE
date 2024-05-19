# NEEDLE Classifier

The NEEDLE classifier is a specialized tool designed for identifying rare astronomical transients from real-time alerts, particularly Superluminous Supernovae (SLSNe) and Tidal Disruption Events (TDEs). Developed by Xinyue Sheng and Matt Nicholl, the NEEDLE classifier uses machine learning models to enhance the detection and classification of these rare events.


![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/github/v/release/XinyueSheng2019/NEEDLE.svg)
![GitHub issues](https://img.shields.io/github/issues/XinyueSheng2019/NEEDLE.svg)
![GitHub forks](https://img.shields.io/github/forks/XinyueSheng2019/NEEDLE.svg)
![GitHub stars](https://img.shields.io/github/stars/XinyueSheng2019/NEEDLE.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/XinyueSheng2019/NEEDLE.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)


## Key Features

- **Authors**: Xinyue Sheng and Matt Nicholl
- **Open Source**: The dataset `data.hdf5` is available for re-training the models.
- **Lasair Version**: Requires an API token, which can be obtained by contacting the Lasair team.
- **Well-trained Models**: Includes three models—NEEDLE-T and NEEDLT-TH—with associated datasets and a Lasair version of the well-trained model (without dataset).
- **Config.py**: A configuration file for inputting file paths.

## Dataset

The dataset is in HDF5 format and is accessible via [this link on Kaggle](https://www.kaggle.com/datasets/sherrysheng97/needle-lasair-dataset).

## Contact Information

For any questions or comments, you can contact Xinyue Sheng at Xsheng03@qub.ac.uk.

## Getting Started

To begin using the NEEDLE classifier, follow these steps:

1. **Download the Dataset**: Download the HDF5 dataset from Kaggle using the provided link.
2. **Set Up Config.py**: Edit the `Config.py` file to include the file paths for your data and models.
3. **Re-train Models**: If desired, use the provided dataset to re-train the models on your local machine.
4. **Request API Token**: For using the Lasair version, request an API token from the Lasair team.

### Example Steps

1. **Download the Dataset**:
   ```sh
   kaggle datasets download -d sherrysheng97/needle-lasair-dataset
