# Final-Project-Machine-Learning-Terapan
Final Project of Machine Learning Terapan Course at Dicoding Indonesia

# Movie Recommendation

## Project Overview (Ulasan Proyek)

![Project Overview](/assets/FinalProject-MLT-ProjectOverview.png "Project Overview")

Film merupakan media audio-visual yang menggabungkan kedua unsur, yaitu naratif dan sinematik [1]. Dalam hal ini, unsur naratif sendiri mempunyai keterikatan dengan tema, sedangkan unsur sinematik merupakan alur atau jalan cerita yang runtun dari awal hingga akhir. Sejak tahun 1874 hingga 2015, sebanyak 3,361,741 judul film telah dikeluarkan oleh industri perfilman [[2](https://citisee.amikompurwokerto.ac.id/assets/proceedings/2017/TI08.pdf)]. Tingkat distribusi film yang begitu masif membuat film menjadi salah satu media hiburan yang populer di kalangan masyarakat luas. Namun, banyaknya judul film yang beredar membuat masyarakat sulit untuk menjumpai film yang sesuai dengan preferensi pengguna Masalah tersebut dapat memicu dampak buruk pada kepuasan dan pengalaman pengguna, seperti kejenuhan yang berpotensi memicu pengguna untuk tidak jadi nonton film. Oleh karena itu, diperlukan suatu sistem yang dapat memberikan rekomendasi film sesuai dengan preferensi pengguna.

Saat ini, perkembangan teknologi informasi yang begitu masif memudahkan setiap orang dalam mencari informasi, tak terkecuali data seputar perfilman. Data-data film yang terdapat pada suatu laman dapat diolah dan dimanfaatkan dalam membangun sistem rekomendasi. Dalam penerapannya, sistem rekomendasi membutuhkan pengetahuan berupa data preferensi pengguna. Kebutuhan tersebut penting untuk dipenuhi guna memberikan rekomendasi item yang sesuai. 

Menurut Paolo Cremonesi dalam course Basic Recommender Systems, sistem rekomendasi dapat diklasifikasikan ke dalam dua kategori besar, yaitu non-personalized (tanpa personalisasi) dan personalized (dengan personalisasi) [[3](https://www.coursera.org/learn/basic-recommender-systems)]. Pada proyek ini, sistem rekomendasi yang dibangun termasuk ke dalam kategori Personalized yang terdiri dari 2 pendekatan, yaitu Content-Based Filtering (CBF) dan Collaborative Filtering (CF).

## Business Uderstanding

### Problem Statements

Berdasarkan latar belakang yang menjadi pembatas bahasan, adapaun rincian masalah yang dapat diselesaikan pada proyek ini yang di antaranya adalah sebagai berikut:
* Bagaimana cara membangun sistem rekomendasi film sesuai genre yang mirip dengan preferensi pengguna di masa lalu dengan menggunakan pendekatan Content-Based Filtering?
* Bagaimana cara membangun sistem rekomendasi film yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya dengan menggunakan pendekatan Collaborative Filtering?

### Goals

Adapun tujuan dibuatnya proyek ini yang di antaranya adalah sebagai berikut:
* Menghasilkan top-N rekomendasi film sesuai genre yang mirip dengan preferensi pengguna di masa lalu dengan menggunakan pendekatan Content-Based Filtering.
* Menghasilkan rekomendasi sejumlah film yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya dengan menggunakan pendekatan Collaborative Filtering.

### Solution Approach

Dalam rangka mencapai goals yang ada, penulis menggunakan 2 pendekatan sistem rekomendasi, yaitu Content-Based Filtering dan Collaborative Filtering. Berikut merupakan penjelasan lebih lengkap dari kedua pendekatan tersebut:  

#### Content-Based Filtering    

Content-Based Filtering merupakan pendekatan sistem rekomendasi yang menggunakan fitur item untuk merekomendasikan item lain yang mirip dengan apa yang disukai pengguna berdasarkan tingkat kesamaannya. Content-Based Filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.

Pada kasus ini, sistem rekomendasi akan memberikan rekomendasi film sesuai dengan genre yang mirip dengan preferensi pengguna di masa lalu. Genre tersebut akan menjadi acuan apakah top-N rekomendasi yang diberikan relevan atau tidak. Dalam hal ini, apabila suatu film mempunyai genre yang mirip, maka film tersebut dapat direkomendasikan kepada user. Berikut merupakan ilustrasinya:

![Content-Based Filtering Illustration](/assets/FinalProject-MLT-Content-BasedFilteringIllustration.png "Content-Based Filtering Illustration")

Dalam implementasinya, penulis menerapkan beberapa tahapan dalam pembentukan model dengan pendekatan Content-Based Filtering. Rincian tahapan tersebut di antaranya adalah sebagai berikut:

1. Assign DataFrame `movie` ke dalam variabel `data`.  
2. Menerapkan proses TF-IDF Vectorizer.  
3. Menerapkan proses Cosine Similarity.
4. Mendapatkan rekomendasi.  

Adapun luaran dari sistem rekomendasi yang dibangun, yaitu berupa 5 rekomendasi film terbaik sesuai dengan preferensi pengguna.

Sementara itu, adapun kekurangan dan kelebihan dari pembuatan model sistem rekomendasi dengan menggunakan pendekatan Content-Based Filtering yang di antaranya adalah sebagai berikut:
1. Kelebihan
  * Mampu menghasilkan top-N rekomendasi yang mirip dengan preferensi pengguna di masa lalu.
  * Tidak bergantung pada pendapat komunitas pengguna, seperti rating karena rekomendasi yang dibuat mengacu pada kemiripan preferensi pengguna sebelumnya.
2. Kekurangan
  * Membutuhkan atribut untuk setiap item.
  
#### Collaborative Filtering

Collaborative Filtering merupakan jenis pendekatan sistem rekomendasi yang membutuhkan data interaksi antar pengguna dalam menentukan daftar rekomendasi. Data tersebut biasanya didapat dari interaksi pengguna di masa sebelumnya. Secara umum, Collaborative Filtering dapat dibagi menjadi 2 kategori, yaitu memory based (berbasis memori) dan model based (berbasis model). Pada kasus ini, penulis menggunakan pendekatan Collaborative Filtering berbasis model.

Dalam implementasinya, penulis menerapkan beberapa tahapan dalam pembentukan model dengan pendekatan Collaborative Filtering. Dalam hal ini, model akan menghitung skor kecocokan antara pengguna dan judul film dengan teknik Embedding. Berikut merupakan rincian tahapan lengkapnya:

1. Melakukan proses embedding terhadap data `user` dan `movie`.
2. Melakukan operasi perkalian dot product antara embedding `user` dan `movie`. Dalam hal ini, bias juga dapat ditambahkan untuk setiap `user` dan `movie`.
3. Skor kecocokan ditetapkan dalam skala [0, 1] dengan fungsi sigmoid.
4. Membangun class `RecommenderNet` dengan keras Model Class [[8](https://keras.io/api/models/model/)]
5. Melakukan proses compile terhadap model.
6. Melakukan proses training.
7. Mendapatkan rekomendasi.

Adapun luaran dari sistem rekomendasi yang dibangun, yaitu berupa 10 rekomendasi film terbaik yang belum ditonton berdasarkan preferensi pengguna (rating) sebelumnya.

Sementara itu, adapun kekurangan dan kelebihan dari pembuatan model sistem rekomendasi dengan menggunakan pendekatan Collaborative Filtering yang di antaranya adalah sebagai berikut:
1. Kelebihan
  * Mampu menghasilkan rekomendasi sejumlah film yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya.
  * Tidak membutuhkan atribut untuk setiap item, seperti pada pendekatan Content-Based Filtering.
2. Kekurangan
  * Bergantung pada pendapat komunitas pengguna. Pada kasus ini, rating.

## Data Understanding

![Dataset Information](/assets/FinalProject-MLT-DatasetInformation.png "Dataset Information")

Dataset yang penulis gunakan pada proyek ini, yaitu dataset dengan judul Movie Recommender System Dataset yang diambil pada laman Kaggle [[4](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset?select=ratings.csv)]. Dataset ini berisikan 2 file bertipe `.csv` dengan judul `movies.csv` dan `ratings.csv`. File `movies.csv` berisikan 9.742 data dengan 3 kolom, sedangkan file ratings.csv berisikan 100.836 dengan 4 kolom. Berikut merupakan informasi lebih detail dari masing-masing kolom pada setiap file yang ada pada pada dataset:
* `movies.csv`
  * `movieId` : ID unik untuk setiap film.
  * `title` : Judul film.
  * `genres` : Genre film, seperti drama, komedi, aksi atau petualang, dan lain sebagainya.

* `ratings.csv`
  * `userId` : ID pengguna yang memberikan peringkat.
  * `movieId` : ID film yang peringkatnya telah diberikan.
  * `rating` : Peringkat yang diberikan oleh pengguna.
  * `timestamp` : Waktu dimana peringkat telah diberikan.
  
### Dataset Information

Kategori | Keterangan
--- | ---
Title | Movie Recommender System Dataset
Source | [Kaggle](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset/metadata?select=ratings.csv)
Maintainer | SHINIGAMI
License | GPL 2
Visibility | Public
Tags | Tabular Data, Recommender Systems
Usability | 8.24

Setelah melakukan analisis pada dataset, penulis mendapatkan informasi yang di antaranya adalah sebagai berikut:
* File `movies.csv` berisikan 9.742 data dengan total 3 kolom yang terdiri dari 1 kolom numerik bertipe data integer (`movieId`) dan 2 kolom kategori bertipe data object (`title` dan `genres`).
* File `ratings.csv` berisikan 100.836 data dengan total 4 kolom bertipe data numerik yang terdiri dari 3 kolom bertipe data integer (`userId`, `movieId`, dan `timestamp`) dan 1 kolom bertipe data float (`rating`).
* Adapun informasi jumlah data yang bersifat unik pada file `movies.csv` yang di antaranya adalah sebagai berikut:
  * Jumlah data `movieId` yang bersifat unik : 9742
  * Jumlah data `title` yang bersifat unik : 9737
  * Jumlah data `genres` yang bersifat unik : 951
* Adapun informasi jumlah data yang bersifat unik pada file `ratings.csv` yang di antaranya adalah sebagai berikut:
  * Jumlah data userId yang bersifat unik : 610
  * Jumlah data movieId yang bersifat unik : 9724
  * Jumlah data rating yang bersifat unik : 10
  * Jumlah data timestamp yang bersifat unik : 85043
* Nilai minimum rating, yakni sebesar 0.5.
* Nilai maximum rating, yakni sebesar 5.0.

## Data Preparation

Data yang sudah dianalisis selanjutnya dipersiapkan dan disesuaikan kembali agar model dapat memahami dan mengolah data. Dalam hal ini, persiapan data dilakukan secara terpisah karena perbedaan kebutuhan sesuai dengan model yang akan dibangun, tepatnya Content-Based Filtering dan Collaborative Filtering. Berikut merupakan penjelasan lengkapnya:

### Content-Based Filtering

Berikut merupakan tahapan persiapan data dari proses pemodelan dengan menggunakan pendekatan Content-Based Filtering:  
1. Melakukan pengecekan missing value pada file `movies.csv`. Tahapan tersebut dilakukan karena penting untuk mempunyai data yang lengkap agar sistem rekomendasi yang dibangun dapat lebih mudah untuk dianalisis. 
2. Melakukan pengecekan data duplikat pada file `movies.csv`. Tahapan tersebut penting untuk dilakukan karena penulis hanya akan menggunakan data yang bersifat unik untuk dimasukkan ke dalam proses pemodelan.

### Collaborative Filtering

Berikut merupakan tahapan persiapan data dari proses pemodelan dengan menggunakan pendekatan Collaborative Filtering:  
1. Melakukan pengecekan missing value pada file `ratings.csv`. Tahapan tersebut dilakukan karena penting untuk mempunyai data yang lengkap agar sistem rekomendasi yang dibangun dapat lebih mudah untuk dianalisis. 
2. Melakukan pengecekan data duplikat pada file `ratings.csv`. Tahapan tersebut penting untuk dilakukan karena penulis hanya akan menggunakan data yang bersifat unik untuk dimasukkan ke dalam proses pemodelan.
3. Menghapus kolom yang tidak dibutuhkan. Dalam hal ini, penulis menghapus kolom yang tidak dibutuhkan, tepatnya kolom `timestamp` pada file `rating.csv` karena tidak berpengaruh pada proses pembuatan model sistem rekomendasi.  
4. Encoding fitur `userId` dan `movieId` agar data dapat diolah saat proses pemodelan. Berikut merupakan rincian tahapan proses Encoding yang diterpakan pada program:  
  * Assign DataFrame `rating` ke dalam variabel `df` guna mempermudah proses persiapan data.
  * Menyandingkan (Encode) fitur `userId` dan `movieId` ke dalam indeks integer.
  * Melakukan pemetaan `userId` dan `movieId` ke DataFrame yang berkaitan.
  * Melihat jumlah dan informasi data yang sudah melewati tahap Encoding
5. Membagi data untuk training dan validasi. Berikut merupakan rincian tahapan proses pembagian data untuk training dan validasi yang diterapkan pada program:  
  * Mengacak dataset agar distribusinya menjadi random.
  * Memetakan (mapping) data user dan movie menjadi satu value.
  * Mengonversi rating ke dalam skala 0 hingga 1 agar mudah melakukan proses training.
  * Membagi data train dan validasi dengan komposisi 80:20.
  
## Modelling and Result

Berdasarkan *Solution Approach* yang menjadi pembatas bahasan, berikut merupakan penjelasan lebih detail dari kedua rincian tahapan modelling dan result dari masing-masing pendekatan: 

### Model Development with Content-Based Filtering

Dengan menggunakan pendekatan Content-Based Filtering, tahapan *modelling* dan *result* akan melalui beberapa tahapan sebagai berikut:
1. Assign DataFrame `movie` ke dalam Variabel `data`.  
2. Menerapkan Proses TF-IDF Vectorizer.  
3. Menerapkan Proses Cosine Similarity.
4. Mendapatkan Rekomendasi.

#### Assign DataFrame `movie` ke dalam Variabel `data`. 

Dalam hal ini, tahapan tersebut penulis lakukan guna mempermudah proses persiapan data.

#### Menerapkan Proses TF-IDF Vectorizer

TF-IDF (Term Frequency-Inverse Document Frequency) merupakan ukuran statistik yang mengevaluasi seberapa relevan sebuah kata dengan dokumen dalam kumpulan dokumen [[5](https://monkeylearn.com/blog/what-is-tf-idf/)]. Ia bertujuan untuk mengukur seberapa penting suatu kata terhadap kata-kata lain dalam dokumen. Secara matematis, TF-IDF didefinisikan dengan dua besaran, yaitu TF dan IDF. TF (Term Frequency) mengukur frekuensi atau seberapa sering suatu kata atau term muncul dalam teks tertentu [6]. Dalam hal ini, teks yang berbeda dalam dokumen berpotensi mempunyai panjang yang berbeda, tergantung dari panjang dokumen. Oleh karena itu, penting untuk melakukan normalisasi dengan membagi jumlah kemunculan terhadap panjang dokumen. Berikut merupakan definisi TF suatu term X pada dokumen d:

![TF Formula](/assets/FinalProject-MLT-TFFormula.jpeg "TF Formula")

Sementara itu, IDF (Inverse Document Frequency) mengukur pentingnya istilah di seluruh korpus [6]. Dalam komputasi TF, seluruh istilah diberikan bobot kepentingan (weight) yang sama, termasuk kata-kata yang tidak penting, seperti stop word (is, are, am, dan lain sebagainya). Dalam hal ini, IDF berfungsi untuk mengatasi kasus tersebut dengan mempertimbangkan term yang sangat umum di seluruh dokumen dan menimbang istilah-istilah yang jarang. IDF untuk suatu term X didefinisikan sebagai berikut:

![IDF Formula](/assets/FinalProject-MLT-IDFFormula.jpeg "IDF Formula")

Guna menghasilkan skor atau bobot untuk TF-IDF, kalikan nilai TF dengan IDF, seperti berikut:

![TF-IDF Formula](/assets/FinalProject-MLT-TF-IDFFormula.jpeg "TF-IDF Formula")

Dalam penerapannya, skor dalam TF-IDF digunakan untuk mengamati istilah-istilah berbeda yang mengandung informasi penting dalam dokumen tertentu. Pada python, perhitungan tersebut dapat diimplementasikan menggunakan fungsi `TfidfVectorizer()` dari library scikit-learn. Dalam hal ini, `TfidfVectorizer()` akan melakukan proses tokenisasi pada teks, mempelajari kosa kata, melakukan pembobotan frekuensi dokumen secara terbalik (inverse), dan memungkinkan pengguna untuk melakukan proses encoding teks baru.Pada kasus ini, TF-IDF Vectorizer penulis gunakan untuk mengidentifikasi korelasi antara judul film (`title`) dengan genre (`genres`).

#### Menerapkan Proses Cosine Similarity

Cosine Similarity adalah metrik yang digunakan untuk menentukan seberapa mirip dokumen terlepas dari ukurannya [[7](https://www.machinelearningplus.com/nlp/cosine-similarity/)]. Dalam penerapannya, cosine similarity akan mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama [6]. Cosine similarity akan menghitung sudut cosinus antara dua vektor dimana semakin kecil sudut cosinus, semakin besar nilai cosine similarity. Berikut merupakan ilustrasinya:

![Cosine Similarity Vector](/assets/FinalProject-MLT-CosineSimilarityVector.jpeg "Cosine Similarity Vector")

Pada proyek ini, cosine similarity digunakan untuk mengukur kesamaan judul film dan genre. Cosine simliarity dapat dirumuskan sebagai berikut:

![Cosine Similarity Formula](/assets/FinalProject-MLT-CosineSimilarityFormula.jpeg "Cosine Similarity Formula")

Pada Python, cosine similarity akan menghitung kesamaan sebagai dot product yang dinormalisasi dari masukan sampel X dan Y. Perhitungan tersebut dapat diimplementasikan dengan menggunakan fungsi `cosine_similarity()` dari library sklearn.

Pada kasus ini, tahap `Cosine Similarity` digunakan untuk menghitung derajat kesamaan antar judul film. Dalam hal ini, penulis menggunakan fungsi `cosine_similarity()` dari library sklearn. Tahap tersebut diterapkan pada DataFrame `tfidf_matrix` yang diperoleh pada tahapan sebelumnya.

#### Mendapatkan Rekomendasi

Selanjutnya, penulis membuat fungsi `movie_recommendations` dengan beberapa parameter berikut:
* `title` : Judul film (index kemiripan DataFrame)
* `similarity_data` : DataFrame mengenai similarity yang telah didefinisikan sebelumnya.
* `items` : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan. Dalam hal ini, yaitu `title` dan `genres`.
* `k` : Banyak rekomendasi yang ingin diberikan.  

Dalam penerapannya, fungsi tersebut penulis gunakan untuk menampilkan 5 rekomendasi film terbaik sesuai dengan preferensi pengguna.

**Important Inferences**
* Dengan menggunakan argpartition, penulis mengambil sejumlah nilai k tertinggi dari similarity data, tepatnya DataFrame `cosine_sim_df`.
* Setelahnya, penulis mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah. Data dimasukkan ke dalam variabel closest.
* Kemudian, penulis menghapus `title` yang dicari agar tidak muncul dalam daftar rekomendasi.

### Model Development with Collaborative Filtering

#### Proses Training

Dengan menggunakan pendekatan Content-Based Filtering, model akan menghitung skor kecocokan antara pengguna dan judul film dengan teknik Embedding. Berikut merupakan rincian tahapan yang terdapat pada program:
1. Melakukan proses embedding terhadap data `user` dan `movie`.
2. Melakukan operasi perkalian dot product antara embedding `user` dan `movie`. Dalam hal ini, bias juga dapat ditambahkan untuk setiap `user` dan `movie`.
3. Skor kecocokan ditetapkan dalam skala [0, 1] dengan fungsi sigmoid.  

Pada kasus ini, penulis membangun class `RecommenderNet` dengan keras Model Class [[8](https://keras.io/api/models/model/)]. Dalam penerapannya, kode program RecommenderNet tersebut terinspirasi dari tutorial dalam situs Keras dengan beberapa adaptasi sesuai kasus proyek [[9](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)].  

Selanjutnya, penulis melakukan proses compile terhadap model. Dalam hal ini, model dibangun dengan menggunakan ketentuan sebagai berikut:
1. Binary Crossentropy untuk menghitung loss function.
2. Adam (Adaptive Moment Estimation) sebagai optimizer.
3. Root Mean Squared Error (RMSE) sebagai metric evaluation.

## Evaluation

### Evaluation of Content-Based Filtering Model

Pada pendekatan Content-Based Filtering, evaluasi dilakukan secara manual menggunakan metric *Precision* (Presisi) yang dikhususkan untuk model sistem rekomendasi. Dalam sistem rekomendasi, *precision* merupakan jumlah item rekomendasi yang relevan dimana penggunaannya dapat dilakukan secara manual. Dalam hal ini, *precision* tidak dapat dihitung dengan memanggil library scikit learn karena tidak ada data target/label, seperti pada *supervised learning*. Pada sistem rekomendasi, precision dapat dirumuskan sebagai berikut:

![Precision Formula](/assets/FinalProject-MLT-PrecisionFormula.png "Precision Formula")

Pada kasus ini, sebelumnya pengguna pernah menonton film `Toy Story (1995)` sehingga sistem akan merekomendasikan film yang similiar dengan film yang pernah ditonton pengguna sebelumnya. Berikut merupakan hasil rekomendasi dari film dengan judul `Toy Story (1995)`.

title | genres
--- | ---
The Good Dinosaur (2015) | adventure, Animation, Children, Comedy, Fantasy
Adventures of Rocky and Bullwinkle, The (2000) | adventure, Animation, Children, Comedy, Fantasy
Moana (2016) | adventure, Animation, Children, Comedy, Fantasy
Wild, The (2006) | adventure, Animation, Children, Comedy, Fantasy
Emperor's New Groove, The (2000) | adventure, Animation, Children, Comedy, Fantasy

Dari hasil rekomendasi di atas, diketahui bahwa `Toy Story (1995)` termasuk ke dalam *genre* `Adventure`, `Animation`, `Children`, `Comedy`, dan `Fantasy`. Dari 5 item yang direkomendasikan, 5 item memiliki genre yang mirip (*similar*) bahkan sama. Oleh karena itu, bila mengacu pada formula *precision* sebelumnya, skor *precision* sistem rekomendasi yang dibuat, yakni sebesar `5/5` atau `100%`.

### Evaluation of Collaborative Filtering Model

Pada pendekatan Collaborative Filtering, evaluasi model dilakukan dengan menggunakan metric Root Mean Squared Error (RMSE). RMSE merupakan metrik standar yang digunakan untuk mengukur kesalahan suatu model dalam memprediksi data kuantitatif [[10](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)]. Dalam penerapannya, semakin kecil skor RMSE, maka akan semakin kecil pula kesalahan prediksi atau rekomendasi suatu model. Secara matematis, RMSE dapat dirumuskan sebagai berikut:

![RMSE Formula](/assets/FinalProject-MLT-RMSEFormula.jpeg "RMSE Formula")

Pada kasus ini, RMSE digunakan untuk mengetahui seberapa jauh model dalam memberikan daftar rekomendasi film dari data sebenarnya. Dari proses ini, penulis memperoleh nilai error akhir sebesar sekitar 0.17 dan error pada data validasi sebesar 0.20. Nilai tersebut menunjukkan bahwa model dapat memberikan daftar rekomendasi dengan kesalahan yang sangat kecil.

#### Visualisasi Metrik

Berikut merupakan proses plotting metrik evaluasi dengan menggunakan library matplotlib.

![Plotting Metric RMSE](/assets/FinalProject-MLT-PlottingMetricRMSE.png "Plotting Metric RMSE")

#### Mendapatkan Rekomendasi

Guna mendapatkan rekomendasi film, penulis menerapkan beberapa tahapan yang di antaranya adalah sebagai berikut:
1. Mengambil sampel user secara acak dan mendefinisikan variabel `movie_not_wathced` yang merupakan daftar film yang belum pernah ditonton oleh pengguna. Sampel tersebut nantinya akan dijadikan film yang akan direkomendasikan oleh sistem.
2. Menggunakan rating film yang telah diberikan pengguna guna membuat rekomendasi film yang mungkin cocok untuk pengguna. Dalam hal ini, rekomendasi yang diberikan berupa rekomendasi film yang belum pernah ditonton pengguna sebelumnya.
3. Dalam penerapannya, variabel `movie_not_watched` diperoleh dengan menggunakan operator bitwise (~) pada variabel `movie_watched_by_user`.

Terakhir, guna memperoleh rekomendasi film, penulis menggunakan fungsi `model.predict()` dari library Keras.

Berikut merupakan hasil rekomendasi dengan menggunakan pendekatan Collaborative FIltering:

![Result Collaborative Filtering](/assets/FinalProject-MLT-ResultCollaborativeFiltering.png "Result Collaborative Filtering")

Hasil di atas merupakan rekomendasi untuk user dengan id 68. Dari output tersebut, dapat disimpulkan bahwa user dapat membandingkan rekomendasi antara **Movie with High Ratings from User** dengan **Top 10 Movie Recommendation**.

## Kesimpulan 

Luaran dari proyek ini, yaitu berupa sistem rekomendasi yang dibangun dengan menggunakan pendekatan Content-Based Filtering dan Collaborative Filtering. Luaran yang dihasilkan dari pemodelan dengan menggunakan pendekatan Content-Based Filtering, yakni 5 rekomendasi film terbaik sesuai dengan preferensi pengguna, sedangkan luaran yang dihasilkan dari pemodelan dengan menggunakan pendekatan Collaborative Filtering, yakni 10 rekomendasi film terbaik yang belum ditonton berdasarkan preferensi pengguna (rating) sebelumnya. Dalam hal ini, model yang dibagun dengan menggunakan pendekatan Content-Based Filtering dievaluasi dengan menggunakan metrik Precision dengan skor sebesar 100%, sedangkan model yang dibangun dengan menggunakan pendekatan Collaborative Filtering dievaluasi dengan menggunakan metrik Root Mean Squared Error (RMSE) dengan memperoleh nilai error akhir sebesar sekitar 0.17 dan error pada data validasi sebesar 0.20. Nilai tersebut menunjukkan bahwa model dapat memberikan daftar rekomendasi dengan kesalahan yang sangat kecil.

## Daftar Referensi

[1] H. Pratista, Memahami Film, Homerian Pustaka, 2008.  
[[2](https://citisee.amikompurwokerto.ac.id/assets/proceedings/2017/TI08.pdf)] A. Halim, H. Gohzali, D. M. Panjaitan, I. Maulana, "Sistem Rekomendasi Film menggunakan Bisecting K-Means dan Collaborative Filtering", 37-41, 2017, https://citisee.amikompurwokerto.ac.id/assets/proceedings/2017/TI08.pdf.  
[[3](https://www.coursera.org/learn/basic-recommender-systems)] P. Cremonesi, "Basic Recommender Systems", https://www.coursera.org/learn/basic-recommender-systems.  
[[4](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset/metadata?select=ratings.csv)] SHINIGAMI, "Movie Recommender System Dataset", Kaggle, 2021. https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset/metadata?select=ratings.csv  
[[5](https://monkeylearn.com/blog/what-is-tf-idf/)] B. Stecanella, "Understanding TF-ID: A Simple Introduction", MonkeyLearn, 2019, https://monkeylearn.com/blog/what-is-tf-idf/.  
[6] Machine Learning Terapan, Dicoding.  
[[7](https://www.machinelearningplus.com/nlp/cosine-similarity/)] S. Prabhakaran, "Cosine Similarity – Understanding the math and how it works (with python codes)", MachineLearing+, 2018. https://www.machinelearningplus.com/nlp/cosine-similarity/  
[[8](https://keras.io/api/models/model/)] Keras, "The Model Class", keras.io. https://keras.io/api/models/model/  
[[9](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)] S. Banerjee, "Collaborative Filtering for Movie Recommendations", keras.io, 2022, https://keras.io/examples/structured_data/collaborative_filtering_movielens/.  
[[10](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)] J. Moody, "What does RMSE really mean?", Medium, 2019, https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e. 
