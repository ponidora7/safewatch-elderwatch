# Buat file test_gpu.py dan jalankan:
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Jika GPU terdeteksi, output akan menampilkan device GPU
# Jika tidak, akan menampilkan list kosong [] - tetap bisa jalan dengan CPU