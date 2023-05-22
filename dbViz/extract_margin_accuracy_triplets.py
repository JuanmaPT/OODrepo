import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

#Added imports TRDP
import sys

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv
from options import options
from utils import simple_lapsed_time

args = options().parse_args()
print(args)

#Fixing the arguments

#Important Note! -> Change the model loading in the intialization
args.load_net = '/home/juanma/TRDP/OODrepo/dbViz/pretrained_models/resnet18-f37072fd.pth'
args.net = 'resnet'
args.set_seed = '777'
args.save_net = 'saves'
args.imgs = 500,5000,1600
args.resolution = 500
args.epochs = 2
args.lr = 0.01
args.resolution = 100 #Default is 500 and it takes 3 mins

# Log of the results
args.active_log = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = args.save_net
if args.active_log:
    import wandb
    idt = '_'.join(list(map(str,args.imgs)))
    wandb.init(project="decision_boundaries", name = '_'.join([args.net,args.train_mode,idt,'seed'+str(args.set_seed)]) )
    wandb.config.update(args)

# Data/other training stuff
torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)
torch.manual_seed(args.set_seed)
test_accs = []
train_accs = []
net = get_model(args, device)

test_acc, predicted = test(args, net, testloader, device, 0)
print("scratch prediction ", test_acc)

criterion = get_loss_function(args)
if args.opt == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = get_scheduler(args, optimizer)

elif args.opt == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)



# Train or load base network
print("Training the network or loading the network")

start = time.time()
best_acc = 0  # best test accuracy
best_epoch = 0

#########################################################
#   TRAINING
#########################################################
if args.load_net is None:
    print("args.load_net is None -> You need to provide the path to the weights!")



#########################################################
#   LOADING THE NETWORK
#########################################################
else:
    net.load_state_dict(torch.load(args.load_net))
    

# test_acc, predicted = test(args, net, testloader, device)
# print(test_acc)
end = time.time()
simple_lapsed_time("Time taken to load the model", end-start)
saveplot = False

if not args.plot_animation:
    from utils import produce_plot_alt,produce_plot_x,produce_plot_sepleg
    start = time.time()
    if args.imgs is None:
        print("args.imgs is None -> You need to provide the images to load")
        
    else:
        # Images that are going to be used for the representation
        #[1 2 3]
        #[1 2 4]
        #[1 2 5]
        #..
        #[2 3 1]
        image_ids = args.imgs
        images = [trainloader.dataset[i][0] for i in image_ids]
        labels = [trainloader.dataset[i][1] for i in image_ids]
        print(labels)

    
    sampleids = '_'.join(list(map(str,image_ids)))
    # sampleids = '_'.join(list(map(str,labels)))

    #Creating planeloader for the image space
    planeloader = make_planeloader(images, args)
    # Extract the image coordinates from the coords variable
    coords_images = planeloader.dataset.coords
    img1_coord = coords_images[0]
    img2_coord = coords_images[1]
    img3_coord = coords_images[2]

    print("Image 1 coordinates:", img1_coord)
    print("Image 2 coordinates:", img2_coord)
    print("Image 3 coordinates:", img3_coord)

    #Using the model to predict all the plane
    preds = decision_boundary(args, net, planeloader, device)
    print("Length of pred(tensors):", len(preds))

    #Getting the labels of the predictions
    preds = torch.stack((preds))
    temp=0.01#Not sure what this does
    preds = nn.Softmax(dim=1)(preds / temp)

    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    x = planeloader.dataset.coefs1.cpu().numpy()
    y = planeloader.dataset.coefs2.cpu().numpy()

    # Mapping the class predictions to a range of 0 to 9
    class_pred = (class_pred % 10).astype(int)

    ## Extracting the predicted labels and indexes of the image triplet
    image_labels = []
    image_indices = []
    for i, image in enumerate(images):
        label = class_pred[i]  # Predicted label for the image
        index = torch.where(planeloader.dataset.coefs1 == torch.min(planeloader.dataset.coefs1))[0][i]  # Index of the image in planeloader
        image_labels.append(label)
        image_indices.append(index)
        print("Predicted Labels:", image_labels)
        print("Image Indices:", image_indices)
    

    net_name = args.net
    if saveplot:
        os.makedirs(f'images/{net_name}/{args.train_mode}/{sampleids}/{str(args.set_seed)}', exist_ok=True)
        #lot_path = os.path.join(args.plot_path,f'{net_name}_{sampleids}_{args.set_seed}cifar10')
        plot_path = os.path.join('paco',f'{net_name}_{sampleids}_{args.set_seed}cifar10')
        os.makedirs(f'{plot_path}', exist_ok=True)
        produce_plot_sepleg(plot_path, preds, planeloader, images, labels, trainloader, title = 'best', temp=1.0,true_labels = None)
        #produce_plot_alt(plot_path, preds, planeloader, images, labels, trainloader)

        # produce_plot_x(plot_path, preds, planeloader, images, labels, trainloader, title=title, temp=1.0,true_labels = None)
    end = time.time()
    simple_lapsed_time("Time taken to plot the image", end-start)
