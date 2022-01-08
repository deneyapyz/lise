import numpy as mat
import skfuzzy as bulanik
from skfuzzy import control as kontrol

bulasik_miktari = kontrol.Antecedent(mat.arange(0, 100, 1), 'bulaşık miktarı')
kirlilik = kontrol.Antecedent( mat.arange(0, 100, 1), 'kirlilik seviyesi')

yikama = kontrol.Consequent(mat.arange(0, 180, 1),'yıkama süresi')


bulasik_miktari['az'] = bulanik.trimf(bulasik_miktari.universe, [0, 0, 30])
bulasik_miktari['normal'] = bulanik.trimf(bulasik_miktari.universe, [10, 30, 60])
bulasik_miktari['çok'] = bulanik.trimf(bulasik_miktari.universe, [50, 60, 100])
kirlilik['az'] = bulanik.trimf(kirlilik.universe, [0, 0, 30])
kirlilik['normal'] = bulanik.trimf(kirlilik.universe, [10, 30, 60])
kirlilik['çok'] = bulanik.trimf(kirlilik.universe, [50, 60, 100])

yikama['kisa'] = bulanik.trimf(yikama.universe, [0, 0, 50])
yikama['normal'] = bulanik.trimf(yikama.universe, [40, 50, 100])
yikama['uzun'] = bulanik.trimf(yikama.universe, [60, 80, 180])


kural1 = kontrol.Rule(bulasik_miktari['az'] & kirlilik['az'], yikama['kisa'])
kural2 = kontrol.Rule(bulasik_miktari['normal'] & kirlilik['az'], yikama['normal'])
kural3 = kontrol.Rule(bulasik_miktari['çok'] & kirlilik['az'], yikama['normal'])
kural4 = kontrol.Rule(bulasik_miktari['az'] & kirlilik['normal'], yikama['normal'])
kural5 = kontrol.Rule(bulasik_miktari['normal'] & kirlilik['normal'], yikama['uzun'])
kural6 = kontrol.Rule(bulasik_miktari['çok'] & kirlilik['normal'], yikama['uzun'])
kural7 = kontrol.Rule(bulasik_miktari['az'] & kirlilik['çok'], yikama['normal'])
kural8 = kontrol.Rule(bulasik_miktari['normal'] & kirlilik['çok'], yikama['uzun'])
kural9 = kontrol.Rule(bulasik_miktari['çok'] & kirlilik['çok'], yikama['uzun'])

sonuc = kontrol.ControlSystem([kural1, kural2, kural3, kural4, kural5, kural6, kural7, kural8, kural9])
model_sonuc = kontrol.ControlSystemSimulation(sonuc) 

model_sonuc.input['bulaşık miktarı'] = 50
model_sonuc.input['kirlilik seviyesi']=80
model_sonuc.compute()
print (model_sonuc.output['yıkama süresi'])



