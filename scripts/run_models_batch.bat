@echo off
REM Runs all three models on pet_images/ and saves benchmark results into categorized results directories

REM Navigate to repository root (parent directory of this script)
cd /d "%~dp0.."

if not exist "results\resnet" mkdir "results\resnet"
if not exist "results\alexnet" mkdir "results\alexnet"
if not exist "results\vgg" mkdir "results\vgg"

echo ========================================================
echo Running Dog Breed Classification Benchmarks (pet_images)
echo ========================================================

echo [1/3] Benchmarking ResNet-18...
python check_images.py --dir pet_images\ --arch resnet --dogfile dognames.txt > results\resnet\resnet_pet-images.txt

echo [2/3] Benchmarking AlexNet...
python check_images.py --dir pet_images\ --arch alexnet --dogfile dognames.txt > results\alexnet\alexnet_pet-images.txt

echo [3/3] Benchmarking VGG-16...
python check_images.py --dir pet_images\ --arch vgg --dogfile dognames.txt > results\vgg\vgg_pet-images.txt

echo Benchmarking complete. Results saved into results\resnet\, results\alexnet\, and results\vgg\.
