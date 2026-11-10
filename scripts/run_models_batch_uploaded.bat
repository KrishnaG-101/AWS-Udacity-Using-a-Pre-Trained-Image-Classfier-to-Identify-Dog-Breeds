@echo off
REM Runs all three models on uploaded_images/ to evaluate custom uploaded image classifications
if not exist "results\resnet" mkdir "results\resnet"
if not exist "results\alexnet" mkdir "results\alexnet"
if not exist "results\vgg" mkdir "results\vgg"

echo =============================================================
echo Running Classification on Uploaded Images (uploaded_images/)
echo =============================================================

echo [1/3] Benchmarking ResNet-18 on uploaded images...
python check_images.py --dir uploaded_images\ --arch resnet --dogfile dognames.txt > results\resnet\resnet_uploaded-images.txt

echo [2/3] Benchmarking AlexNet on uploaded images...
python check_images.py --dir uploaded_images\ --arch alexnet --dogfile dognames.txt > results\alexnet\alexnet_uploaded-images.txt

echo [3/3] Benchmarking VGG-16 on uploaded images...
python check_images.py --dir uploaded_images\ --arch vgg --dogfile dognames.txt > results\vgg\vgg_uploaded-images.txt

echo Benchmarking complete. Results saved into results\resnet\, results\alexnet\, and results\vgg\.
