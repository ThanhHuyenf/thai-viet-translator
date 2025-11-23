# Script để tạo virtual environment, cài thư viện và chạy training
# Thai-Vietnamese Translation Model

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "THAI-VIETNAMESE TRANSLATOR - SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Tạo virtual environment
Write-Host "1. Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "   Virtual environment already exists, skipping..." -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "   ✓ Virtual environment created" -ForegroundColor Green
}
Write-Host ""

# 2. Kích hoạt virtual environment
Write-Host "2. Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "   ✓ Virtual environment activated" -ForegroundColor Green
Write-Host ""

# 3. Upgrade pip
Write-Host "3. Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "   ✓ Pip upgraded" -ForegroundColor Green
Write-Host ""

# 4. Cài đặt thư viện
Write-Host "4. Installing dependencies..." -ForegroundColor Yellow
Write-Host "   Installing PyTorch..." -ForegroundColor Gray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
Write-Host ""
Write-Host "   Installing PyTorch Geometric..." -ForegroundColor Gray
pip install torch-geometric
Write-Host ""
Write-Host "   Installing other dependencies..." -ForegroundColor Gray
pip install pandas numpy
Write-Host "   ✓ All dependencies installed" -ForegroundColor Green
Write-Host ""

# 5. Tạo thư mục cần thiết
Write-Host "5. Creating necessary directories..." -ForegroundColor Yellow
if (-not (Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints" | Out-Null
}
if (-not (Test-Path "vocab")) {
    New-Item -ItemType Directory -Path "vocab" | Out-Null
}
Write-Host "   ✓ Directories created" -ForegroundColor Green
Write-Host ""

# 6. Hiển thị thông tin
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CONFIGURATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Data split: 80% train, 10% val, 10% test" -ForegroundColor White
Write-Host "Model: GCN + LSTM Encoder -> Shared Gates LSTM Decoder" -ForegroundColor White
Write-Host "Batch size: 64" -ForegroundColor White
Write-Host "Epochs: 30" -ForegroundColor White
Write-Host "Dictionary size: ~13,000 words" -ForegroundColor White
Write-Host ""

# 7. Chạy training
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STARTING TRAINING" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop training..." -ForegroundColor Yellow
Write-Host ""

python train.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Models saved in: checkpoints/" -ForegroundColor White
Write-Host "Vocabularies saved in: vocab/" -ForegroundColor White
Write-Host ""
Write-Host "To run inference:" -ForegroundColor Yellow
Write-Host "  python inference.py --interactive" -ForegroundColor White
Write-Host "  python inference.py --word <thai_word>" -ForegroundColor White
Write-Host ""
