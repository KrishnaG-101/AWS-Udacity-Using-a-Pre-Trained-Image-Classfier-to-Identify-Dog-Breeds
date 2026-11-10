# PowerShell batch script to benchmark all 3 CNN architectures on pet_images/
$null = New-Item -ItemType Directory -Force -Path "results/resnet", "results/alexnet", "results/vgg"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Running Dog Breed Classification Benchmarks (pet_images)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Benchmarking ResNet-18..." -ForegroundColor Yellow
python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt | Tee-Object -FilePath "results/resnet/resnet_pet-images.txt"

Write-Host "`n[2/3] Benchmarking AlexNet..." -ForegroundColor Yellow
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt | Tee-Object -FilePath "results/alexnet/alexnet_pet-images.txt"

Write-Host "`n[3/3] Benchmarking VGG-16..." -ForegroundColor Yellow
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt | Tee-Object -FilePath "results/vgg/vgg_pet-images.txt"

Write-Host "`nBenchmarking complete. Results saved to results/resnet/, results/alexnet/, and results/vgg/" -ForegroundColor Green
