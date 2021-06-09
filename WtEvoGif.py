import torch
import imageio
import os
import matplotlib.pyplot as plt 

#Define pathname of stored images
EvoPath = os.path.join(os.getcwd(),'Weight evolution')

#Need to load all weights and save their plots as some sort of image. 
filenames = [] #A list of all of the filenames for the gif

for i in range(10000):
    #Load and generate plot for just the weights.
    filestring = '%dImages.pt' %(i+1)
    weights = torch.load(os.path.join(EvoPath,filestring))

    plt.matshow(weights,fignum=1,cmap='hot_r',vmin=-45e-3,vmax=45e-3)
    imgname = os.path.join(EvoPath,'%d.png' %(i))
    plt.savefig(imgname)
    plt.cla()
    filenames.append(imgname)
    print('%d out of 10,000\n'%(i))



#Build the Gif using imageio

with imageio.get_writer('WtEvoGif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)




