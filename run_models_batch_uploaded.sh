#!/bin/sh
# Runs all three models on uploaded_images/ and saves benchmark results into categorized results directories
mkdir -p results/resnet results/alexnet results/vgg

python check_images.py --dir uploaded_images/ --arch resnet --dogfile dognames.txt > results/resnet/resnet_uploaded-images.txt
python check_images.py --dir uploaded_images/ --arch alexnet --dogfile dognames.txt > results/alexnet/alexnet_uploaded-images.txt
python check_images.py --dir uploaded_images/ --arch vgg --dogfile dognames.txt > results/vgg/vgg_uploaded-images.txt
