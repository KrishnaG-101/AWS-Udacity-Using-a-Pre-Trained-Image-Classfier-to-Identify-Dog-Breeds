# PowerShell batch script to benchmark all 3 CNN architectures on uploaded_images/
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "Running Classification on Uploaded Images (uploaded_images/)" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Benchmarking ResNet-18 on uploaded images..." -ForegroundColor Yellow
python check_images.py --dir uploaded_images/ --arch resnet --dogfile dognames.txt | Tee-Object -FilePath resnet_uploaded-images.txt

Write-Host "`n[2/3] Benchmarking AlexNet on uploaded images..." -ForegroundColor Yellow
python check_images.py --dir uploaded_images/ --arch alexnet --dogfile dognames.txt | Tee-Object -FilePath alexnet_uploaded-images.txt

Write-Host "`n[3/3] Benchmarking VGG-16 on uploaded images..." -ForegroundColor Yellow
python check_images.py --dir uploaded_images/ --arch vgg --dogfile dognames.txt | Tee-Object -FilePath vgg_uploaded-images.txt

Write-Host "`nBenchmarking complete. Results saved to resnet_uploaded-images.txt, alexnet_uploaded-images.txt, and vgg_uploaded-images.txt" -ForegroundColor Green
