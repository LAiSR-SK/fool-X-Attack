#test our approach and deepfool and FGSM
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from ImmunityTestingFunction import hybridImmunityTesting
from ImmunityTestingFunction import deepfoolImmunityTesting
from ImmunityTestingFunction import FGSMImmunityTestingImproved
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
from foolx import foolx
import os
import glob
import cv2

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

#Evaluation of immunity on deepfool, hybrid, FGSM; finetunes network on adversarial examples than uses functions in ImmunityTestingFunction.py to test

#Define the network to be finetuned and use to train
#net = models.resnet34(pretrained=True)
#net = models.alexnet(pretrained=True)
net_resnet101 = models.resnet101(pretrained=True)
#net = models.googlenet(pretrained=True)
#Put network on GPU
net_resnet101.cuda()
#Set network to evaluation mode
net_resnet101.eval()
#net2 = models.resnet34(pretrained=True)
#net2.cuda()
#net2.eval()
#net_alexnet = models.alexnet(pretrained=True)
#net_alexnet.cuda()
#net_alexnet.eval()

ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

#Define networks to be finetuned for each approach, load them into the GPU, and set them all in training mode
#deepfoolnet = models.resnet101(pretrained=True)
#deepfoolnet.cuda()
#deepfoolnet.train()

#hybridnet = models.resnet101(pretrained=True)
#hybridnet.cuda()
#hybridnet.train()

#fgsmnet = models.resnet101(pretrained=True)
#fgsmnet.cuda()
#fgsmnet.train()

#net_resnet34 = models.resnet34(pretrained=True)
#net_resnet34.cuda()
#net_resnet34.eval()


#deepfoolnet_alex = models.alexnet(pretrained=True)
#deepfoolnet_alex.cuda()
#deepfoolnet_alex.train()

#hybridnet_alex_0005 = models.alexnet(pretrained=True)
#hybridnet_alex_0005.cuda()
#hybridnet_alex_0005.train()

#hybridnet_alex_05 = models.alexnet(pretrained=True)
#hybridnet_alex_05.cuda()
#hybridnet_alex_05.train()

#hybridnet_alex_2 = models.alexnet(pretrained=True)
#hybridnet_alex_2.cuda()
#hybridnet_alex_2.train()

hybridnet_resnet101_2 = models.resnet101(pretrained=True)
hybridnet_resnet101_2.cuda()
hybridnet_resnet101_2.train()

#fgsmnet_alex_0005 = models.alexnet(pretrained=True)
#fgsmnet_alex_0005.cuda()
#fgsmnet_alex_0005.train()

#fgsmnet_alex_05 = models.alexnet(pretrained=True)
#fgsmnet_alex_05.cuda()
#fgsmnet_alex_05.train()

#fgsmnet_alex_2 = models.alexnet(pretrained=True)
#fgsmnet_alex_2.cuda()
#fgsmnet_alex_2.train()

#deepfool_alex = models.alexnet(pretrained=True)
#deepfool_alex.cuda()
#deepfool_alex.train()

#fgsmnet_res34_0005 = models.resnet34(pretrained=True)
#fgsmnet_res34_0005.cuda()
#fgsmnet_res34_0005.train()

#fgsmnet_res34_05 = models.resnet34(pretrained=True)
#fgsmnet_res34_05.cuda()
#fgsmnet_res34_05.train()

#fgsmnet_res34_2 = models.resnet34(pretrained=True)
#fgsmnet_res34_2.cuda()
#fgsmnet_res34_2.train()

#fgsmnet_res101_0005 = models.resnet101(pretrained=True)
#fgsmnet_res101_0005.cuda()
#fgsmnet_res101_0005.train()

#fgsmnet_res101_05 = models.resnet101(pretrained=True)
#fgsmnet_res101_05.cuda()
#fgsmnet_res101_05.train()

#fgsmnet_res101_2 = models.resnet101(pretrained=True)
#fgsmnet_res101_2.cuda()
#fgsmnet_res101_2.train()


#Training function for deepfool, takes the original network, the network to be finetuned, and the name of the network that will be used in the csv and pth files
def trainDeepfoolImmunity(orig_net, immunenet, name):
    immunenet.cuda()
    immunenet.train()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #set criterion as cross entropy loss with adam optimizer, use a very low learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(immunenet.parameters(), lr=1e-5)

    for epoch in range(5):  # loop over the dataset multiple times
        if epoch != 0:
            #Save csv and pth file every epoch
            PATH = './deepfooladv_net_' + str(epoch) + '.pth'
            torch.save(immunenet.state_dict(), PATH)
            immunenet.eval()
            csv = 'deepfool' + name + 'immunityepoch' + str(epoch) + '.csv'
            deepfoolImmunityTesting(orig_net, immunenet, csv)
            immunenet.train()
        running_loss = 0.0
        i = 0
        counter = 0
        for filename in glob.glob('D:/imageNet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
            if counter == 5000:
                break
            im_orig = Image.open(filename).convert('RGB')
            im = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)])(im_orig)
            r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool(im, orig_net)
            # get the inputs; data is a list of [inputs, labels]
            # inputs = data[None, :, :, :]
            inputs = pert_image
            labels = ILSVRClabels[np.int(counter)].split(' ')[1]
            labels = torch.tensor([int(labels)])
            labels = labels.to('cuda')
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = immunenet(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            i = i + 1
            counter = counter + 1
    print('Finished Training')
    PATH = './deepfooladv_net_' + name + '.pth'
    torch.save(immunenet.state_dict(), PATH)
    return immunenet

#Training function for hybrid, takes the original network, the network to be finetuned, the epsilon value, and the name of the network that will be used in the csv and pth files
def trainHybridImmunity(orig_net, immunenet, name, eps):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(immunenet.parameters(), lr=1e-5)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for epoch in range(5):  # loop over the dataset multiple times
        if epoch != 0:
            PATH = './hybridadv_net' + name + str(epoch) + '.pth'
            torch.save(immunenet.state_dict(), PATH)
            immunenet.eval()
            csv = 'hybrid' + name + 'immunityepoch' + str(epoch) + str(eps) + '.csv'
            hybridImmunityTesting(orig_net, immunenet, eps, csv)
            immunenet.train()
        running_loss = 0.0
        i = 0
        counter = 0
        for filename in glob.glob('D:/imageNet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
            if counter == 5000:
                break
            im_orig = Image.open(filename).convert('RGB')
            im = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)])(im_orig)
            r, loop_i, label_orig, label_pert, pert_image, newf_k = foolx(im, orig_net, eps)
            inputs = pert_image
            labels = ILSVRClabels[np.int(counter)].split(' ')[1]
            labels = torch.tensor([int(labels)])
            labels = labels.to('cuda')
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = immunenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            i = i + 1
            counter = counter + 1
    print('Finished Training')
    PATH = './hybridadv_net' + str(eps) + name + '.pth'
    torch.save(immunenet.state_dict(), PATH)
    return immunenet

#Training function for FGSM, takes the original network, the network to be finetuned, the epsilon value, and the name of the network that will be used in the csv and pth files
def trainFGSMImmunity(orig_net, immunenet, name, eps):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(immunenet.parameters(), lr=1e-5)

    for epoch in range(5):  # loop over the dataset multiple times
        if epoch != 0:
            PATH = './fgsmadv_net' + name + str(eps) + str(epoch) + '.pth'
            torch.save(immunenet.state_dict(), PATH)
            immunenet.eval()
            csv = 'fgsm' + name + 'immunityepoch' + str(eps) + str(epoch) + '.csv'
            FGSMImmunityTestingImproved(orig_net, immunenet, eps, csv)
            immunenet.train()
        running_loss = 0.0
        i = 0
        counter = 0
        for filename in glob.glob('D:/imageNet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
            # get the inputs; data is a list of [inputs, labels]
            if counter == 5000:
                break
            orig = cv2.imread(filename)[..., ::-1]
            orig = cv2.resize(orig, (224, 224))
            img = orig.copy().astype(np.float32)
            perturbation = np.empty_like(orig)

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img /= 255.0
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)

            inp = Variable(torch.from_numpy(img).to('cuda:0').float().unsqueeze(0), requires_grad=True)
            out = orig_net(inp)
            pred = np.argmax(out.data.cpu().numpy())

            loss = criterion(out, Variable(torch.Tensor([float(pred)]).to('cuda:0').long()))
            loss.backward()

            inp.data = inp.data + (eps * torch.sign(inp.grad.data))
            print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
            inp.grad.data.zero_()  # unnecessary
            inputs = inp
            labels = ILSVRClabels[np.int(counter)].split(' ')[1]
            #print(labels)
            labels = torch.tensor([int(labels)])
            labels = labels.to('cuda')
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = immunenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            i = i + 1
            counter = counter + 1

    print('Finished Training')
    #Save adversarial net
    PATH = './fgsmadv_net_' + str(eps) + name + '.pth'
    torch.save(immunenet.state_dict(), PATH)
    return immunenet

def trainFGSMImmunityImproved(orig_net, immunenet, name, eps):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(immunenet.parameters(), lr=1e-5)

    for epoch in range(5):  # loop over the dataset multiple times
        if epoch != 0:
            PATH = './fgsmadv_net' + name + str(eps) + str(epoch) + '.pth'
            torch.save(immunenet.state_dict(), PATH)
            immunenet.eval()
            csv = 'fgsm' + name + 'immunityepoch' + str(eps) + str(epoch) + '.csv'
            FGSMImmunityTestingImproved(orig_net, immunenet, eps, csv)
            immunenet.train()
        running_loss = 0.0
        i = 0
        counter = 0
        for filename in glob.glob('D:/imageNet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
            # get the inputs; data is a list of [inputs, labels]
            if counter == 5000:
                break

            im_orig = Image.open(filename).convert('RGB')

            im = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)])(im_orig)

            input_batch = im.unsqueeze(0)
            input_array = input_batch.numpy()
            inp = torch.autograd.Variable(torch.from_numpy(input_array[0]).to('cuda:0').float().unsqueeze(0),
                                          requires_grad=True)
            out = orig_net(inp)
            pred = np.argmax(out.data.cpu().numpy())

            loss = criterion(out, Variable(torch.Tensor([float(pred)]).to('cuda:0').long()))
            loss.backward()

            inp.data = inp.data + (eps * torch.sign(inp.grad.data))
            print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
            inp.grad.data.zero_()  # unnecessary
            inputs = inp
            labels = ILSVRClabels[np.int(counter)].split(' ')[1]
            #print(labels)
            labels = torch.tensor([int(labels)])
            labels = labels.to('cuda')
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = immunenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            i = i + 1
            counter = counter + 1

    print('Finished Training')
    #Save adversarial net
    PATH = './fgsmadv_net_' + str(eps) + name + '.pth'
    torch.save(immunenet.state_dict(), PATH)
    return immunenet


#Call functions for training and testing
#deepfoolnet = trainDeepfoolImmunity(net, deepfoolnet, 'resnet101')
#hybridnet = trainHybridImmunity(net, hybridnet, 'resnet101', 0.2)
#fgsmnresnet1010005 = trainFGSMImmunityImproved(net, fgsmnet, 'resnet101', 0.0005)
#deepfoolnet.eval()
#hybridnet.eval()
#fgsmnresnet1010005.eval()
#deepfoolImmunityTesting(net, deepfoolnet, 'deepfoolresnet101immunityfinished.csv')
#hybridImmunityTesting(net, hybridnet, 0.2, 'hybridresnet101immunityfinished0.2.csv')
#FGSMImmunityTestingImproved(net, fgsmnresnet1010005, 0.0005, 'fgsmresnet101immunityfinished0.0005.csv')


#fgsmresnet1012 = trainFGSMImmunityImproved(net_resnet101, fgsmnet_res101_2, 'resnet101', 0.2)
#fgsmresnet1012.eval()
#FGSMImmunityTestingImproved(net_resnet101, fgsmresnet1012, 0.2, 'fgsmresnet101ImmunityFinished0.2.csv')

#fgsmresnet1010005 = trainFGSMImmunityImproved(net_resnet101, fgsmnet_res101_0005, 'resnet101', 0.0005)
#fgsmresnet1010005.eval()
#FGSMImmunityTestingImproved(net_resnet101, fgsmresnet1010005, 0.0005, 'fgsmresnet101ImmunityFinished0.0005.csv')

#fgsmresnet10105 = trainFGSMImmunityImproved(net_resnet101, fgsmnet_res101_05, 'resnet101', 0.05)
#fgsmresnet10105.eval()
#FGSMImmunityTestingImproved(net_resnet101, fgsmresnet10105, 0.05, 'fgsmresnet101ImmunityFinished0.05.csv')

#fgsmresnet342 = trainFGSMImmunityImproved(net_resnet34, fgsmnet_res34_2, 'resnet34', 0.2)
#fgsmresnet342.eval()
#FGSMImmunityTestingImproved(net_resnet34, fgsmresnet342, 0.2, 'fgsmresnet34ImmunityFinished0.2.csv')

#fgsmresnet340005 = trainFGSMImmunityImproved(net_resnet34, fgsmnet_res34_0005, 'resnet34', 0.0005)
#fgsmresnet340005.eval()
#FGSMImmunityTestingImproved(net_resnet34, fgsmresnet340005, 0.0005, 'fgsmresnet34ImmunityFinished0.0005.csv')

#fgsmresnet3405 = trainFGSMImmunityImproved(net_resnet34, fgsmnet_res34_05, 'resnet34', 0.05)
#fgsmresnet3405.eval()
#FGSMImmunityTestingImproved(net_resnet34, fgsmresnet3405, 0.05, 'fgsmresnet34ImmunityFinished0.05.csv')

#fgsmalexnet2 = trainFGSMImmunityImproved(net_alexnet, fgsmnet_alex_2, 'alexnet', 0.2)
#fgsmalexnet2.eval()
#FGSMImmunityTestingImproved(net_alexnet, fgsmalexnet2, 0.2, 'fgsmalexnetImmunityFinished0.2.csv')

#fgsmalexnet0005 = trainFGSMImmunityImproved(net_alexnet, fgsmnet_alex_0005, 'alexnet', 0.0005)
#fgsmalexnet0005.eval()
#FGSMImmunityTestingImproved(net_alexnet, fgsmalexnet0005, 0.0005, 'fgsmalexnetImmunityFinished0.0005.csv')

#fgsmalexnet05 = trainFGSMImmunityImproved(net_alexnet, fgsmnet_alex_05, 'alexnet', 0.05)
#fgsmalexnet05.eval()
#FGSMImmunityTestingImproved(net_alexnet, fgsmalexnet05, 0.05, 'fgsmalexnetImmunityFinished0.05.csv')

hybridresnet1012 = trainHybridImmunity(net_resnet101, hybridnet_resnet101_2, 'resnet101', 0.2)
hybridresnet1012.eval()
hybridImmunityTesting(net_resnet101, hybridresnet1012, 0.2, 'hybridresnet101ImmunityFinished0.2.csv')



