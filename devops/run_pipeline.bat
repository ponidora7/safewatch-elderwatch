@echo off
title SafeWatch Command Center
color 0B

:MENU
cls
echo =======================================================
echo         🛡️ SAFEWATCH AI - COMMAND CENTER 🛡️
echo =======================================================
echo.
echo Pilih modul yang ingin dijalankan:
echo [1] 🧠 Latih Ulang Model AI (train_model.py)
echo [2] 📊 Nyalakan Dasbor Analitik (Streamlit)
echo [3] 🚀 Aktifkan Kamera Pengawas Real-Time (run_safewatch.py)
echo [4] ❌ Keluar
echo.
echo =======================================================
set /p pilihan="Masukkan angka pilihanmu [1/2/3/4]: "

if "%pilihan%"=="1" goto LatihModel
if "%pilihan%"=="2" goto BukaDasbor
if "%pilihan%"=="3" goto NyalakanKamera
if "%pilihan%"=="4" goto Keluar

:: Jika salah ketik
echo Pilihan tidak valid! Silakan coba lagi.
timeout /t 2 >nul
goto MENU

:LatihModel
cls
echo Memulai proses pelatihan ulang model kecerdasan buatan...
echo.
python scripts\train_model.py
echo.
pause
goto MENU

:BukaDasbor
cls
echo Menyiapkan server lokal untuk Dasbor Streamlit...
echo Tekan CTRL+C di terminal ini nanti jika ingin mematikan server.
echo.
streamlit run dashboard\app.py
echo.
pause
goto MENU

:NyalakanKamera
cls
echo Memuat arsitektur Hibrida YOLO dan MediaPipe...
echo.
python scripts\run_safewatch.py
echo.
pause
goto MENU

:Keluar
cls
echo Mematikan Command Center... Selamat beristirahat, Kapten!
timeout /t 2 >nul
exit