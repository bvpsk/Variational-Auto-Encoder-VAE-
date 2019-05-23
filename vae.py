###############################################################################################################################
#
#	Variational Auto Encoder Built using Keras and celebA dataset is used
#	for training.
#
#	Created by : B V P Sai Kumar
#	github : https://github.com/kumararduino
#	website : https://kumarbasaveswara.in
#	linkedin : https://www.linkedin.com/in/kumar15412304/
#
#
#	Credits:
#	Dataset : https://www.kaggle.com/jessicali9530/celeba-dataset
#	Article : https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
#		This Article gave me a clear glance about how Variational Auto Encoders work
# 	Article on KL_Divergence : https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained
#	Trained this model on 'Google Colab'
#	Link for Google Colab : https://colab.research.google.com/drive/19TVLOcllujpiBh3y2EUUQXwgdyUSwBv_
#
#
#
###############################################################################################################################



# Importing necessary packages
import argparse
import numpy as np
import multiprocessing as mp
import numpy as np
import cv2
import keras
from keras.models import Model,Sequential,load_model
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Activation,Input,Lambda,Flatten,Reshape,Conv2DTranspose
import keras.backend as K
from random import randint as r


def vae_loss(input_img, output):
	# compute the average MSE error, then scale it up, ie. simply sum on all axes
	reconstruction_loss = K.sum(K.square(output-input_img))
	# compute the KL loss
	kl_loss = - 0.5 * K.sum(1 + sd_layer - K.square(mean_layer) - K.square(K.exp(sd_layer)), axis=-1)
	# return the average loss over all images in batch
	total_loss = K.mean(reconstruction_loss + kl_loss)
	return total_loss


def sampler(layers):
  std_norm = K.random_normal(shape=(K.shape(layers[0])[0], 128), mean=0, stddev=1)
  return layers[0] + layers[1]*std_norm


ap = argparse.ArgumentParser()
ap.add_argument("--weights","-w",required = True)
ap.add_argument("--generate","-g",required = True,type = int,default = 0)#pass 1 as CommandLine argument to generate faces
ap.add_argument("--image","-i",required = False)
args = vars(ap.parse_args())

#These are mean and standard deviation values obtained from the celebA dataset used for training
mean = 0.43810788
std = 0.29190385

if args["generate"] == 1:
    img = np.random.normal(size = (9,32,32,3))
else:
    img = cv2.imread(args["image"])
    print(img.shape)
    img = cv2.resize(img,(32,32),interpolation = cv2.INTER_AREA)
    img = img.astype("float32")/255.0
    img = (img - mean)/std


#Building the Variational Auto Encoder

stride = 2
#Building the Encoder
inp = Input(shape = (32,32,3)) #using a 32,32,3 image as an input
x = inp
x = Conv2D(32,(2,2),strides = stride,activation = "relu",padding = "same")(x)
x = Conv2D(64,(2,2),strides = stride,activation = "relu",padding = "same")(x)
x = Conv2D(128,(2,2),strides = stride,activation = "relu",padding = "same")(x)
shape = K.int_shape(x)
x = Flatten()(x)
x = Dense(256,activation = "relu")(x)
mean_layer = Dense(128,activation = "relu")(x)
sd_layer = Dense(128,activation = "relu")(x)
latent_vector = Lambda(sampler)([mean_layer,sd_layer])
encoder = Model(inp,latent_vector,name = "VAE_Encoder")
#Encoder is built

#Building the decoder
decoder_inp = Input(shape = (128,))
x = decoder_inp
x = Dense(shape[1]*shape[2]*shape[3],activation = "relu")(x)
x = Reshape((shape[1],shape[2],shape[3]))(x)
x = (Conv2DTranspose(32,(3,3),strides = stride,activation = "relu",padding = "same"))(x)
x = (Conv2DTranspose(16,(3,3),strides = stride,activation = "relu",padding = "same"))(x)
x = (Conv2DTranspose(8,(3,3),strides = stride,activation = "relu",padding = "same"))(x)
outputs = Conv2DTranspose(3, (3,3), activation = 'sigmoid', padding = 'same', name = 'decoder_output')(x)
decoder = Model(decoder_inp,outputs,name = "VAE_Decoder")
#Decoder is built

#Connecting Encoder and decoder
autoencoder = Model(inp,decoder(encoder(inp)),name = "Variational_Auto_Encoder")

autoencoder.load_weights(args["weights"]) #Loading pre-trained weights
autoencoder.compile(optimizer = "adam",loss = vae_loss,metrics = ["accuracy"])




if args["generate"] == 1:
    pred = autoencoder.predict(img)
    op = np.vstack((np.hstack((pred[0],pred[1],pred[2])),np.hstack((pred[3],pred[4],pred[5])),np.hstack((pred[6],pred[7],pred[8]))))
    print(op.shape)
    op = cv2.resize(op,(288,288),interpolation = cv2.INTER_AREA)
    cv2.imshow("generated",op)
    cv2.imwrite("generated"+str(r(0,9999))+".jpg",(op*255).astype("uint8"))
else:
    pred = autoencoder.predict(img.reshape(1,32,32,3))
    cv2.imshow("prediction",cv2.resize(pred[0],(96,96),interpolation = cv2.INTER_AREA))
while cv2.waitKey(0) != 27:
    pass
cv2.destroyAllWindows()
