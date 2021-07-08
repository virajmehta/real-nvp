import torch
import matplotlib.pyplot as plt
import numpy as np

def list2bar(arr, x):
    max_value = np.max(enc_var_bar)
    min_value = np.min(enc_var_bar)
    max_x = np.max(x)
    min_x = np.min(x)
    assert max_value <= max_x
    assert min_value >= min_x
    heights = []
    for item in x:
        heights.append(np.sum(arr == item))
    return heights

vae_var_folder = ["plain_VAE_mnist_var_256/", "plain_VAE_mnist_var_128/", "plain_VAE_mnist_var_64/"]
latent_dims = [256, 128, 64]

fig, axes = plt.subplots(nrows=1, ncols=len(vae_var_folder), figsize=(5 * len(vae_var_folder), 5))

for i in range(len(vae_var_folder)):
    name = vae_var_folder[i]
    logvar = torch.load("data/" + name + "logvar99.pt", map_location=torch.device('cpu'))
    output_var = torch.load("data/" + name + "output_var99.pt", map_location=torch.device('cpu'))
    
    print ("encoder variance")
    print (torch.exp(logvar[-1]))
    print ("decoder variance")
    print (output_var[-1])
    x = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2]
    enc_var_bar = np.sort(np.round(torch.exp(logvar[-1][0, 0, :]).detach().numpy(), decimals=1))
    enc_var_heights = list2bar(enc_var_bar, x)
    axes[i].bar(x, enc_var_heights, width = 0.1)
    axes[i].set_title("Latent Dimension " + str(latent_dims[i]))
    axes[i].set_ylim((0, 140))

#plt.savefig("plots_rnvp/variance_var.png")



