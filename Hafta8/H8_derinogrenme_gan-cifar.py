from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
 
# ayrımcıyı model tanımlama
def ayrimci(in_shape=(32,32,3)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 
# üretici model tanımlama
def uretici(son_boyut):
	model = Sequential()
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=son_boyut))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model
 
# üretici ve ayrımcı modellerin birleştirme 
def GANs(g_model, d_model):
	d_model.trainable = False
	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
 
# veri seti yükleme
def Veri_set_yukleme():
	(trainX, _), (_, _) = load_data()
	X = trainX.astype('float32')
	X = (X - 127.5) / 127.5 #[-1,1]
	return X
 
# örnek veri seçme
def ornek_veri(dataset, n_samples):
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = ones((n_samples, 1))
	return X, y
 
# gizli noktalar oluşturmak
def noktalar_olusturma(son_boyut, n_samples):
	x_input = randn(son_boyut * n_samples)
	x_input = x_input.reshape(n_samples, son_boyut)
	return x_input
 
#sahte nesneler oluşturma
def sahte_nesne_olusturma(g_model, son_boyut, n_samples):
	x_input = noktalar_olusturma(son_boyut, n_samples)
	X = g_model.predict(x_input)
	y = zeros((n_samples, 1))
	return X, y
 
# çizim
def cizim(examples, epoch, n=7):
	examples = (examples + 1) / 2.0
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i])
	filename = 'oluşturulan_ornek %03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()
 
# model değerlendirme
def degerlendir(epoch, g_model, d_model, dataset, son_boyut, n_samples=150):
	X_gercek, y_gercek = ornek_veri(dataset, n_samples)
	_, acc_gercek = d_model.evaluate(X_gercek, y_gercek, verbose=0)
	x_sahte, y_sahte = sahte_nesne_olusturma(g_model, son_boyut, n_samples)
	_, acc_sahte = d_model.evaluate(x_sahte, y_sahte, verbose=0)
	print('gerçek doğruluğu %.0f%%, sahte doğruluğu: %.0f%%' % (acc_gercek*100, acc_sahte*100))
	cizim(x_sahte, epoch)
	filename = 'oluşturulan_model_%03d.h5' % (epoch+1)
	g_model.save(filename)
 
# model eğitimi
def train(g_model, d_model, gan_model, dataset, son_boyut, n_epochs=30, n_batch=128):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	for i in range(n_epochs):
		for j in range(bat_per_epo):
			X_gercek, y_gercek = ornek_veri(dataset, half_batch)
			d_loss1, _ = d_model.train_on_batch(X_gercek, y_gercek)
			X_sahte, y_sahte = sahte_nesne_olusturma(g_model, son_boyut, half_batch)
			d_loss2, _ = d_model.train_on_batch(X_sahte, y_sahte)
			X_gan = noktalar_olusturma(son_boyut, n_batch)
			y_gan = ones((n_batch, 1))
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			print('>%d, %d/%d, dr=%.3f, ds=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
		if (i+1) % 5 == 0:
			degerlendir(i, g_model, d_model, dataset, son_boyut)
 
son_boyut = 100
d_model = ayrimci()
g_model = uretici(son_boyut)
gan_model = GANs(g_model, d_model)
dataset = Veri_set_yukleme()
train(g_model, d_model, gan_model, dataset, son_boyut)