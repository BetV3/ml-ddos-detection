#!/bin/bash
echo "Downloading the dataset..."
curl -L -o ../data/raw/network-intrusion-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset

echo "Unzipping the dataset..."
unzip ../data/raw/network-intrusion-dataset.zip -d ../data/raw/

echo "Removing the zip file..."
rm ../data/raw/network-intrusion-dataset.zip

echo "Done!"
