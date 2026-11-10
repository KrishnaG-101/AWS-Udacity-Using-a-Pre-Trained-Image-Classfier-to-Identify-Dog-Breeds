#!/bin/sh
# Runs all three models on pet_images/ and saves benchmark results into categorized results directories
mkdir -p results/resnet results/alexnet results/vgg

python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt > results/resnet/resnet_pet-images.txt
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > results/alexnet/alexnet_pet-images.txt
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt > results/vgg/vgg_pet-images.txt
