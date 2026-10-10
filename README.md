# Proyek Akhir: Analisis Attrition Karyawan

## Business Understanding

Attrition adalah kondisi ketika karyawan meninggalkan perusahaan. Proyek ini menganalisis pola attrition pada data karyawan dan menyediakan dashboard eksploratif untuk membantu HR menentukan area yang perlu ditinjau lebih lanjut. Hasil analisis bersifat deskriptif dan tidak membuktikan hubungan sebab-akibat.

### Permasalahan Bisnis

- Berapa tingkat attrition pada karyawan yang memiliki label?
- Faktor pekerjaan, kepuasan, dan demografi mana yang menunjukkan asosiasi paling kuat dengan attrition?
- Bagaimana HR dapat membandingkan kelompok karyawan tanpa mengabaikan ukuran sampel?
- Sebanyak 412 baris tidak memiliki label attrition, sehingga cakupan analisis target terbatas.

### Cakupan Proyek

- Memeriksa kualitas, tipe, nilai hilang, dan distribusi data.
- Mengeksplorasi seluruh fitur dan mengukur asosiasinya dengan attrition sesuai tipe data.
- Membandingkan kelompok faktor melalui dashboard Streamlit.
- Menampilkan kontribusi faktor menggunakan regresi logistik, validasi silang 5-fold, dan permutation importance.
- Tidak membuat keputusan HR otomatis atau menyimpulkan sebab-akibat.

### Persiapan

Sumber data: [Employee Data dari Dicoding](https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/employee/employee_data.csv). Dashboard membaca file `employee_data_cleaned.csv` yang tersedia di folder proyek.

Setup environment (PowerShell):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Business Dashboard

Dashboard menampilkan jumlah karyawan berlabel, jumlah attrition, dan tingkat attrition. Pengguna dapat membandingkan tingkat attrition menurut faktor seperti lembur, jabatan, departemen, kepuasan kerja, dan masa kerja, serta mengatur jumlah minimum karyawan per kelompok. Dashboard juga menampilkan ROC-AUC validasi silang dan permutation importance dari model regresi logistik.

Jalankan aplikasi dengan perintah di atas, lalu buka [http://localhost:8501](http://localhost:8501). Dashboard belum dipublikasikan sebagai layanan daring.

## Conclusion

Dataset memiliki 1.470 baris dan 35 kolom. Sebanyak 1.058 baris memiliki label attrition; 179 di antaranya berlabel attrition, sehingga tingkat attrition pada data berlabel adalah **16,9%**. Tidak ditemukan baris duplikat. `EmployeeId` merupakan identitas, sementara `EmployeeCount`, `Over18`, dan `StandardHours` bernilai konstan sehingga tidak informatif sebagai faktor pembeda.

Asosiasi numerik terkuat terlihat pada total pengalaman kerja (median 7 tahun untuk kelompok attrition dan 10 tahun untuk kelompok non-attrition), pendapatan bulanan (3.388 dan 5.210), serta masa kerja di perusahaan (3 dan 6 tahun). Di antara faktor kategorikal, `OverTime` dan `JobRole` menunjukkan asosiasi tertinggi. Korelasi kuat antara pendapatan dan level jabatan (Spearman 0,926) serta antarvariabel masa kerja menunjukkan bahwa beberapa fitur membawa informasi yang tumpang tindih.

### Rekomendasi Action Items (Optional)

- Tinjau beban lembur dan pola attrition per jabatan bersama HR. Gunakan hasil dashboard sebagai sinyal untuk investigasi, bukan bukti bahwa lembur menyebabkan karyawan keluar.
- Lakukan wawancara retensi dan tinjau jalur pengembangan karier pada kelompok dengan masa kerja atau pengalaman lebih rendah; validasi temuan sebelum menetapkan kebijakan.
- Audit 412 baris tanpa label dan perbaiki pencatatan status karyawan. Jangan menganggap label yang hilang sebagai “tidak attrition” atau mengisinya tanpa verifikasi.
- Saat membangun model, pertimbangkan korelasi antarfitur dan validasi performa pada data baru sebelum hasilnya digunakan untuk mendukung keputusan HR.

### URL Dashboard
https://employees-attrition-analysis.streamlit.app