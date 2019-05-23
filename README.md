# Variational-Auto-Encoder-VAE-
Implementing VAE in keras and training on CelebA dataset

This is implementing Variational Auto Encoder in keras

vae.py is the python file containing the code to run VAE

It can be operated in two modes

<h1>1)Transforming Custom Image</h1>
  Here,We pass an image and we get the output of the autoencoder when that image is passed as an input
  
  To perform this,the command line Syntax is as follows:
  <br>
  

    python vae.py -i <IMAGE PATH> -w <WEIGHTS FILE PATH> -g 1
  
  <br>
    
<h1>2)Generating a new face</h1>
  We can generate a new face by the following command ine syntax
  <br>
  
    python vae.py -w <WEIGHTS FILE PATH> -g 1
  
  see this in action
  
  until now,this VAE is partially trained.In future,I will commit this repo with my further work to improve this model.
  
  
<h1>Generative Outputs</h1>

  <h3>Date : 23/05/2019</h3>
  
  <img src = "https://github.com/kumararduino/Variational-Auto-Encoder-VAE-/blob/master/generated3148.jpg"  />  <img src = "https://github.com/kumararduino/Variational-Auto-Encoder-VAE-/blob/master/generated9262.jpg"/>
