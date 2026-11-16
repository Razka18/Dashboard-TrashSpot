"""PENGATURAN PATH TRASHSPOT — edit file ini saja.

CARA MENGISI PATH
-----------------
1. Di Windows, salin path folder dari File Explorer (klik kanan folder ->
   "Copy as path"), lalu tempel di antara r"..." seperti contoh di bawah.
   Awalan huruf r penting agar tanda \\ (backslash) tidak error.
       folder: r"D:\\Proyek\\TrashSpot\\CCTV_1_Jalan_Utama"
2. Boleh juga path RELATIF terhadap folder proyek (folder tempat app.py berada):
       folder: r"CCTV_1_Jalan_Utama"
3. Kosongkan ("") untuk deteksi otomatis folder bernama CCTV_<nomor>_... di folder
   proyek. Jika folder tidak ada / tidak berisi video, kotak tampil offline
   ("Tidak ada sinyal") dan pesan penyebabnya muncul di atas halaman.
4. Isi folder = kumpulan video (.mp4, .avi, .mov, .mkv). Video diputar berurutan
   menurut nama (vid0011.mp4, vid0027.mp4, ...) lalu mengulang dari awal.
5. Simpan file ini, lalu muat ulang halaman (tekan R di browser).

label  = nama lokasi di kartu & riwayat      banjar = nama banjar/dusun
Tambah/hapus kamera dengan menambah/menghapus baris nomor di CAMERAS.
"""

CAMERAS = {
    1: {"label": "Jalan Utama", "banjar": "[Nama banjar]",
        "folder": r"CCTV_1_Jalan_Utama"},
    2: {"label": "Taman dan Ruang Terbuka", "banjar": "[Nama banjar]",
        "folder": r"CCTV_2_Taman_dan_Ruang_Hijau"},                      # kosong = cari otomatis CCTV_2_...
    3: {"label": "Sungai", "banjar": "[Nama banjar]",
        "folder": r"PERLU_VERIFIKASI_Tidak_Sesuai_Kategori"},                      # kosong & tidak ada folder = offline
    4: {"label": "Tempat Sampah", "banjar": "[Nama banjar]",
        "folder": r"CCTV_4_Tempat_Sampah"},
}

# Path model YOLO (.pt). Kosongkan ("") untuk mencari otomatis di folder MODEL/.
MODEL_PATH = r"MODEL/best (2).pt"