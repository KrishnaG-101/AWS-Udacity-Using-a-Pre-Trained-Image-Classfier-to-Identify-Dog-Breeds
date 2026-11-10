@echo off
REM Runs all three models on pet_images/ to benchmark and compare architecture results
echo ========================================================
echo Running Dog Breed Classification Benchmarks (pet_images)
echo ========================================================

echo [1/3] Benchmarking ResNet-18...
python check_images.py --dir pet_images\ --arch resnet --dogfile dognames.txt > resnet_pet-images.txt

echo [2/3] Benchmarking AlexNet...
python check_images.py --dir pet_images\ --arch alexnet --dogfile dognames.txt > alexnet_pet-images.txt

echo [3/3] Benchmarking VGG-16...
python check_images.py --dir pet_images\ --arch vgg --dogfile dognames.txt > vgg_pet-images.txt

echo Benchmarking complete. Results written to resnet_pet-images.txt, alexnet_pet-images.txt, and vgg_pet-images.txt.
