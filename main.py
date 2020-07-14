from keras.utils.vis_utils import plot_model
from utils import *

if __name__ == "__main__":
    latent_dim = 100
    d_model = discriminator_arch()
    g_model = generator_arch(latent_dim)
    gan_model = GAN_arch(g_model, d_model)
    dataset = load_real_samples()
    train(g_model, d_model, gan_model, dataset, latent_dim)

