import numpy as np
ortam_satir_sayisi = 11
ortam_sutun_sayisi = 11
q_degerleri = np.zeros((ortam_satir_sayisi, ortam_sutun_sayisi, 4))
hareketler = ['yukari', 'sag', 'asagi', 'sol']
oduller = np.full((ortam_satir_sayisi, ortam_sutun_sayisi), -100.)
oduller[0, 5] = 100.
gecitler = {} 
gecitler[1] = [i for i in range(1, 10)]
gecitler[2] = [1, 7, 9]
gecitler[3] = [i for i in range(1, 8)]
gecitler[3].append(9)
gecitler[4] = [3, 7]
gecitler[5] = [i for i in range(11)]
gecitler[6] = [5]
gecitler[7] = [i for i in range(1, 10)]
gecitler[8] = [3, 7]
gecitler[9] = [i for i in range(11)]

for satir_indeks in range(1, 10):
  for sutun_indeks in gecitler[satir_indeks]:
    oduller[satir_indeks, sutun_indeks] = -1.

for satir in oduller:
  print(satir)

def engel_mi(gecerli_satir_indeks, gecerli_sutun_indeks):
  if oduller[gecerli_satir_indeks, gecerli_sutun_indeks] == -1.:
    return False
  else:
    return True

def baslangic_belirle():
  gecerli_satir_indeks = np.random.randint(ortam_satir_sayisi)
  gecerli_sutun_indeks = np.random.randint(ortam_sutun_sayisi)
  while engel_mi(gecerli_satir_indeks, gecerli_sutun_indeks):
    gecerli_satir_indeks = np.random.randint(ortam_satir_sayisi)
    gecerli_sutun_indeks = np.random.randint(ortam_sutun_sayisi)
  return gecerli_satir_indeks, gecerli_sutun_indeks

def sonraki_hareket_belirle(gecerli_satir_indeks, gecerli_sutun_indeks, epsilon):
  if np.random.random() < epsilon:
    return np.argmax(q_degerleri[gecerli_satir_indeks, gecerli_sutun_indeks])
  else: 
    return np.random.randint(4)

def sonraki_noktaya_git(gecerli_satir_indeks, gecerli_sutun_indeks, hareket_indeks):
  yeni_satir_indeks = gecerli_satir_indeks
  yeni_sutun_indeks = gecerli_sutun_indeks
  if hareketler[hareket_indeks] == 'yukari' and gecerli_satir_indeks > 0:
    yeni_satir_indeks -= 1
  elif hareketler[hareket_indeks] == 'sag' and gecerli_sutun_indeks < ortam_sutun_sayisi - 1:
    yeni_sutun_indeks += 1
  elif hareketler[hareket_indeks] == 'asagi' and gecerli_satir_indeks < ortam_satir_sayisi - 1:
    yeni_satir_indeks += 1
  elif hareketler[hareket_indeks] == 'sol' and gecerli_sutun_indeks > 0:
    yeni_sutun_indeks -= 1
  return yeni_satir_indeks, yeni_sutun_indeks

def en_kisa_mesafe(basla_satir_indeks, basla_sutun_indeks):
  if engel_mi(basla_satir_indeks, basla_sutun_indeks):
    return []
  else:
    gecerli_satir_indeks, gecerli_sutun_indeks = basla_satir_indeks, basla_sutun_indeks
    en_kisa = []
    en_kisa.append([gecerli_satir_indeks, gecerli_sutun_indeks])
    while not engel_mi(gecerli_satir_indeks, gecerli_sutun_indeks):
      hareket_indeks = sonraki_hareket_belirle(gecerli_satir_indeks, gecerli_sutun_indeks, 1.)
      gecerli_satir_indeks, gecerli_sutun_indeks = sonraki_noktaya_git(gecerli_satir_indeks, 
                                                                       gecerli_sutun_indeks, hareket_indeks)
      en_kisa.append([gecerli_satir_indeks, gecerli_sutun_indeks])
    return en_kisa

epsilon = 0.9
azalma_degeri = 0.9 
ogrenme_orani = 0.9 

for adim in range(1000):
  satir_indeks, sutun_indeks = baslangic_belirle()
  while not engel_mi(satir_indeks, sutun_indeks):
    hareket_indeks = sonraki_hareket_belirle(satir_indeks, sutun_indeks, epsilon)
    eski_satir_indeks, eski_sutun_indeks = satir_indeks, sutun_indeks
    satir_indeks, sutun_indeks = sonraki_noktaya_git(satir_indeks, sutun_indeks, hareket_indeks)
    odul = oduller[satir_indeks, sutun_indeks]
    eski_q_degeri = q_degerleri[eski_satir_indeks, eski_sutun_indeks, hareket_indeks]
    fark = odul + (azalma_degeri * np.max(q_degerleri[satir_indeks, sutun_indeks])) - eski_q_degeri
    yeni_q_degeri = eski_q_degeri + (ogrenme_orani * fark)
    q_degerleri[eski_satir_indeks, eski_sutun_indeks, hareket_indeks] = yeni_q_degeri
print('Eğitim tamamlandı.')

ogr_sonrasi_satir = input('Robotun harekete başlayacağı satır indeksini giriniz:')
ogr_sonrasi_sutun = input('Robotun harekete başlayacağı satır indeksini giriniz:')

print('Kargo noktasına giden rota:',
      en_kisa_mesafe(int(ogr_sonrasi_satir), int(ogr_sonrasi_sutun)))
