# Deep Learning Course Projects

Bu depo, YZM304 Derin Ogrenme dersi boyunca yapilan proje odevlerini tek yerde toplamak icin kullaniliyor.

## Projects
- `project1/`: BankNote Authentication veri seti uzerinde scratch MLP ve PyTorch karsilastirmasi
- `project2/`: CIFAR-10 uzerinde custom CNN, ResNet18 ve hibrit CNN+SVM karsilastirmasi
- `project4/`: Turkce toksik dil tespiti icin TF-IDF baseline, transformer karsilastirmasi, kalibrasyon, threshold politikalari, detayli rapor ve PowerPoint sunum
- `project5/`: OCRTurk uzerinde Turkce OCR, diakritik hata analizi ve IEEE tarzinda derleme/deneysel on calisma bildirisi

## Current Outputs

- Project1 ve Project2: deney kodlari, sonuclar ve grafikler kendi klasorlerinde yer alir.
- Project4: final sunum dosyasi `project4/output/Project4_Turkish_Toxic_Language_Detection.pptx` altindadir.
- Project5: mail eki olarak kullanilacak PDF bildiri `project5/output/pdf/Project5_OCRTurk_IEEE_Format_Report.pdf` altindadir.
- Project5 icin OCRTurk ham verisi, render edilen sayfalar, smoke/demo sonuclari ve ornek bildiri PDF'leri repoda tutulmaz.

## Usage
Her proje kendi klasoru icinde bagimsiz calisacak sekilde duzenlenmistir. Ilgili proje klasorune girip kendi `README.md` dosyasindaki talimatlari izleyebilirsiniz.

## GPU Setup
Windows genel CUDA ve GPU destekli PyTorch kurulumu icin [docs/windows_gpu_setup.md](docs/windows_gpu_setup.md) dosyasina bakin.
