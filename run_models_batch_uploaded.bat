@echo off
REM Runs all three models on uploaded_images/ to evaluate custom uploaded image classifications
echo =============================================================
echo Running Classification on Uploaded Images (uploaded_images/)
echo =============================================================

echo [1/3] Benchmarking ResNet-18 on uploaded images...
python check_images.py --dir uploaded_images\ --arch resnet --dogfile dognames.txt > resnet_uploaded-images.txt

echo [2/3] Benchmarking AlexNet on uploaded images...
python check_images.py --dir uploaded_images\ --arch alexnet --dogfile dognames.txt > alexnet_uploaded-images.txt

echo [3/3] Benchmarking VGG-16 on uploaded images...
python check_images.py --dir uploaded_images\ --arch vgg --dogfile dognames.txt > vgg_uploaded-images.txt

echo Benchmarking complete. Results written to resnet_uploaded-images.txt, alexnet_uploaded-images.txt, and vgg_uploaded-images.txt.
