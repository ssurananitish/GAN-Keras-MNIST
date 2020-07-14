from numpy import expand_dims, ones, zeros, vstack
from numpy.random import randint, rand, randn
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, LeakyReLU, Conv2DTranspose, Reshape
from keras.optimizers import Adam
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt

def vis_images(dataset):
    for i in range(15):
        plt.subplot(5,3,i+1)
        plt.axis('off')
        plt.imshow(dataset[i],cmap='gray_r')
    plt.show()

def discriminator_arch(in_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding="same",input_shape = in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3,3), strides=(2,2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    
    optimizer = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def load_real_samples():
    (trainX, _), (_,_) = load_data()
    trainX = expand_dims(trainX, axis=-1)
    trainX = trainX.astype('float32')
    trainX = trainX/255.0
    return trainX

def generated_real_samples(dataset, n_samples):
    i = randint(0,dataset.shape[0], n_samples)
    sample = dataset[i]
    label = ones((n_samples,1))
    return sample, label

def generated_fake_samples(g_model, latent_dim, n_samples):
    fake_sample = generated_latent_points(latent_dim, n_samples)
    prediction = g_model.predict(fake_sample)
    label = zeros((n_samples, 1))
    return prediction, label

def train_discriminator(model, dataset, n_iter=100,n_batch=256):
    half_batch = int(n_batch/2)
    for i in range(n_iter):
        X_real, y_real = generated_real_samples(dataset, half_batch)
        _, real_accuracy = model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generated_fake_samples(half_batch)
        _, fake_accuracy = model.train_on_batch(X_fake, y_fake)
        print('{}: real={}% \t fake={}%'.format(i+1, real_accuracy*100, fake_accuracy*100))

def generator_arch(latent_dim):
    model = Sequential()
    n_nodes = 128*7*7
    model.add(Dense(n_nodes, input_dim = latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7,7,128)))
    model.add(Conv2DTranspose(128,(4,4), strides=(2,2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128,(4,4), strides=(2,2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7),activation="sigmoid", padding="same"))
    return model

def generated_latent_points(latent_dim, n_samples):
    point = randn(latent_dim * n_samples)
    point = point.reshape(n_samples, latent_dim)
    return point

def GAN_arch(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    return model

def train_GAN(gan_model, latent_dim, n_batch=256, n_epochs=100):
    for i in range(n_epochs):
        x_gan = generated_latent_points(latent_dim,n_batch)
        y_gan = ones((n_batch,1))
        gan_model.train_on_batch(x_gan,y_gan)

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    bat_per_epoch = int(dataset.shape[0]/ n_batch)
    half_batch = int(n_batch/2)
    for i in range(n_epochs):
        for j in range(bat_per_epoch):
            X_real, y_real = generated_real_samples(dataset, half_batch)
            X_fake, y_fake = generated_fake_samples(g_model, latent_dim, half_batch)
            X,y = vstack((X_real,X_fake)), vstack((y_real,y_fake))
            d_loss, _ = d_model.train_on_batch(X,y)
            X_gan = generated_latent_points(latent_dim,n_batch)
            y_gan = ones((n_batch,1))
            g_loss = gan_model.train_on_batch(X_gan,y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epoch, d_loss, g_loss))

        if(i+1)%10==0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples = 100):
    X_real, y_real = generated_real_samples(dataset, n_samples)
    _, real_accuracy = d_model.evaluate(X_real, y_real, verbose = 0)
    X_fake, y_fake = generated_fake_samples(g_model, latent_dim, n_samples)
    _, fake_accuracy = d_model.evaluate(X_fake, y_fake, verbose = 0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (real_accuracy*100, fake_accuracy*100))
    save_plot(X_fake, epoch)
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)

def save_plot(examples, epoch, n=10):
    for i in range(n*n):
        plt.subplot(n,n,i+1)
        plt.axis('off')
        plt.imshow(examples[i,:,:,0], c_map="gray_r")
    
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()



