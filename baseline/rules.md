# ATURAN BASELINE

## Pembuatan
- Baseline dibuat SEKALI sebelum eksperimen pertama dimulai.
- Baseline mencakup seluruh file di folder `baseline/`.
- Engineer yang membuat baseline bertanggung jawab atas kelengkapan data.

## Kekekalan
- Baseline TIDAK BOLEH DIUBAH setelah dibuat.
- Jika ditemukan kesalahan, buat baseline baru dengan ID berbeda, dan catat alasannya.

## Perbandingan
- Semua eksperimen (Phase 5) WAJIB dibandingkan dengan baseline ini.
- Setiap eksperimen harus mencantumkan Baseline ID di dokumen eksperimen.

## Versi Baru
- Jika model di-retrain atau dataset berubah, buat baseline baru (BL-002, dst.).
- Baseline lama TETAP DISIMPAN sebagai arsip.

## Kelengkapan
- Baseline dianggap SAH jika semua file (BASELINE.md, metrics.json, model_info.json, dataset_info.json, config_snapshot.json, environment.json) lengkap dan terisi.
- Jika satu file kosong atau tidak ada, baseline BELUM SAH.