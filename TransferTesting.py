import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
from PIL import Image
from deepfool import deepfool
from foolx import deepfool_hybrid2
from models.AlexNet import AlexNet
import os
import time
import glob
from imagenet_labels import classes
from torch.autograd import Variable
import torch
import torch.nn as nn
import cv2
import csv

#Transferability testing, generates perturbations on one network architecture, tests on another

orig_alexnet = AlexNet()
state = torch.load("models/alexnet/model.pth")
orig_alexnet.load_state_dict(state)
#orig_resnet34 = models.resnet34(pretrained=False)
#state = torch.load("models/resnet/model.pth")
#orig_resnet34.load_state_dict(state)
#eval_googlenet = models.googlenet(pretrained=True)
#eval_alexnet = AlexNet()
#state = torch.load("models/resnet/model.pth")
#eval_alexnet.load_state_dict(state)
eval_resnet34 = models.resnet34(pretrained=False)
state = torch.load("models/resnet/model.pth")
eval_resnet34.load_state_dict(state)

#Transfer testing for hybrid, select first architecture to generate perturbations with, select second architecture to test against, epsilon value, and csv name
def TransferTestingHybrid(net, net2, eps, csvname):
    net.cuda()
    net2.cuda()
    net.eval()
    net2.eval()
    hybridApproach_Testing_Results = ""
    Net1Accuracy = 0
    Net1Hybrid2Accuracy = 0
    Net2Accuracy = 0
    Net2Hybrid2Accuracy = 0
    hybridcsv = csvname
    fieldnames = ['Image', 'Correct Label', 'Network 1 Orig Label', 'Network 2 Orig Label', 'Network 1 Pert Label',
              'Network 2 Pert Label']

    hybrows = []
    counter = 0

    with open(hybridcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')
    for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        if counter == 5000:
            break
        print(" \n\n\n**************** Hybrid Approach DeepFool 2 *********************\n")
        im_orig = Image.open(filename).convert('RGB')
        print(filename)
        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])(im_orig)
        r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool_hybrid2(im, net)
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]

        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

        str_label_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_pert = labels[np.int(label_pert)].split(',')[0]
        str_label_correct = labels[np.int(correct)].split(',')[0]

        print("Correct label = ", str_label_correct)
        print("Network 1 Original label = ", str_label_orig)
        print("Network 1 Perturbed label = ", str_label_pert)

        if (int(label_orig) == int(correct)):

            print("Network 1 Classifier is correct on original image")
            Net1Accuracy = Net1Accuracy + 1

        if (int(label_pert) == int(correct)):
            print("Network 1 Classifier is correct on perturbed image")
            Net1Hybrid2Accuracy = Net1Hybrid2Accuracy + 1

        label2 = net2(im[None, :].cuda())
        label2 = np.argmax(label2.detach().cpu().numpy())
        str_label_orig2 = labels[np.int(label2)].split(',')[0]
        label_pert2 = net2(pert_image)
        label_pert2 = np.argmax(label_pert2.detach().cpu().numpy())
        str_label_pert2 = labels[np.int(label_pert2)].split(',')[0]

        print("Network 2 Original label = ", str_label_orig2)
        print("Network 2 Perturbed label = ", str_label_pert2)

        if (int(label2) == int(correct)):
            print("Network 2 Classifier is correct on original image")
            Net2Accuracy = Net2Accuracy + 1

        if (int(label_pert2) == int(correct)):
            print("Network 2 Classifier is correct on perturbed image")
            Net2Hybrid2Accuracy = Net2Hybrid2Accuracy + 1

        hybrows = []
        hybrows.append(
            [filename[47:75], str_label_correct, str_label_orig, str_label_orig2, str_label_pert, str_label_pert2])
        with open(hybridcsv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(hybrows)

        print("Iterations: " + str(loop_i))
        counter = counter + 1

    with open(hybridcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Original Accuracy: " + str(Net1Accuracy / 5000)])
        csvwriter.writerows(["Original Perturbed Accuracy: " + str(Net1Hybrid2Accuracy / 5000)])
        csvwriter.writerows(["Transfered Accuracy: " + str(Net2Accuracy / 5000)])
        csvwriter.writerows(["Transferred Perturbed Accuracy: " + str(Net2Hybrid2Accuracy / 5000)])
        csvwriter.writerows(['Original Network: ' + 'resnet34'])
        csvwriter.writerows(['Transfered Network: ' + 'resnet101'])

#Transfer testing for deepfool, select first architecture to generate perturbations with, select second architecture to test against and csv name
def TransferTestingDeepfool(net, net2, csvname):
    net.cuda()
    net2.cuda()
    net.eval()
    net2.eval()
    Net1Accuracy = 0
    Net1DeepfoolAccuracy = 0
    Net2Accuracy = 0
    Net2DeepfoolAccuracy = 0
    deepfoolcsv = csvname
    fieldnames = ['Image', 'Correct Label', 'Network 1 Orig Label', 'Network 2 Orig Label', 'Network 1 Pert Label',
                  'Network 2 Pert Label']

    counter = 0

    with open(deepfoolcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')
    for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        if counter == 5000:
            break
        print(" \n\n\n**************** DeepFool *********************\n")
        im_orig = Image.open(filename).convert('RGB')
        print(filename)
        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])(im_orig)
        r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool(im, net)
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]

        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

        str_label_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_pert = labels[np.int(label_pert)].split(',')[0]
        str_label_correct = labels[np.int(correct)].split(',')[0]

        print("Correct label = ", str_label_correct)
        print("Network 1 Original label = ", str_label_orig)
        print("Network 1 Perturbed label = ", str_label_pert)

        if (int(label_orig) == int(correct)):
            print("Network 1 Classifier is correct on original image")
            Net1Accuracy = Net1Accuracy + 1

        if (int(label_pert) == int(correct)):
            print("Network 1 Classifier is correct on perturbed image")
            Net1DeepfoolAccuracy = Net1DeepfoolAccuracy + 1

        label2 = net2(im[None, :].cuda())
        label2 = np.argmax(label2.detach().cpu().numpy())
        str_label_orig2 = labels[np.int(label2)].split(',')[0]
        label_pert2 = net2(pert_image)
        label_pert2 = np.argmax(label_pert2.detach().cpu().numpy())
        str_label_pert2 = labels[np.int(label_pert2)].split(',')[0]

        print("Network 2 Original label = ", str_label_orig2)
        print("Network 2 Perturbed label = ", str_label_pert2)

        if (int(label2) == int(correct)):
            print("Network 2 Classifier is correct on original image")
            Net2Accuracy = Net2Accuracy + 1

        if (int(label_pert2) == int(correct)):
            print("Network 2 Classifier is correct on perturbed image")
            Net2DeepfoolAccuracy = Net2DeepfoolAccuracy + 1

        dfrows = []
        dfrows.append(
            [filename[47:75], str_label_correct, str_label_orig, str_label_orig2, str_label_pert, str_label_pert2])
        with open(deepfoolcsv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(dfrows)

        print("Iterations: " + str(loop_i))
        counter = counter + 1

    with open(deepfoolcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Original Accuracy: " + str(Net1Accuracy / 5000)])
        csvwriter.writerows(["Original Perturbed Accuracy: " + str(Net1DeepfoolAccuracy / 5000)])
        csvwriter.writerows(["Transfered Accuracy: " + str(Net2Accuracy / 5000)])
        csvwriter.writerows(["Transfered Perturbed Accuracy: " + str(Net2DeepfoolAccuracy / 5000)])
        csvwriter.writerows(['Original Network: ' + 'resnet34'])
        csvwriter.writerows(['Transfered Network: ' + 'resnet101'])

#Transfer testing for FGSM, select first architecture to generate perturbations with, select second architecture to test against, epsilon value, and csv name
def TransferTestingFGSM(net, net2, eps, csvname):
    net.cuda()
    net2.cuda()
    net.eval()
    net2.eval()
    Net1Accuracy = 0
    Net2Accuracy = 0
    Net1FGSMAccuracy = 0
    Net2FGSMAccuracy = 0
    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')
    fgsmcsv = csvname
    fieldnames = ['Image', 'Correct Label', 'Network 1 Orig Label', 'Network 2 Orig Label', 'Network 1 Pert Label',
                  'Network 2 Pert Label']
    eps = eps
    counter = 0
    with open(fgsmcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)
    for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        if counter == 5000:
            break
        print('FGSM Testing')
        orig = cv2.imread(filename)[..., ::-1]
        print(filename)
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]
        orig = cv2.resize(orig, (224, 224))
        img = orig.copy().astype(np.float32)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img /= 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)

        inp = Variable(torch.from_numpy(img).to('cuda:0').float().unsqueeze(0), requires_grad=True)

        out = net(inp)
        out2 = net2(inp)
        criterion = nn.CrossEntropyLoss()
        pred = np.argmax(out.data.cpu().numpy())
        pred2 = np.argmax(out2.data.cpu().numpy())
        loss = criterion(out, Variable(torch.Tensor([float(pred)]).to('cuda:0').long()))
        loss.backward()
        print('Network 1 Prediction before attack: %s' % (classes[pred].split(',')[0]))
        print('Network 2 Prediction before attack: %s' % (classes[pred2].split(',')[0]))
        if (int(pred) == int(correct)):
            print("Network 1 Classifier is correct")
            Net1Accuracy = Net1Accuracy + 1
        if (int(pred2) == int(correct)):
            print("Network 2 Classifier is correct")
            Net2Accuracy = Net2Accuracy + 1

        # this is it, this is the method
        inp.data = inp.data + (eps * torch.sign(inp.grad.data))
        inp.grad.data.zero_()  # unnecessary

        # predict on the adversarial image
        pred_adv = np.argmax(net(inp).data.cpu().numpy())
        pred_adv2 = np.argmax(net2(inp).data.cpu().numpy())
        print("After attack on original network: " + str(eps) + " " + classes[pred_adv].split(',')[0])
        print("After attack on transferred network: " + str(eps) + " " + classes[pred_adv2].split(',')[0])
        if (int(pred_adv) == int(correct)):
            print("Network 1 Classifier is correct on perturbed image")
            Net1FGSMAccuracy = Net1FGSMAccuracy + 1
        if (int(pred_adv2) == int(correct)):
            print("Network 2 Classifier is correct on perturbed image")
            Net2FGSMAccuracy = Net2FGSMAccuracy + 1
        # deprocess image
        adv = inp.data.cpu().numpy()[0]
        perturbation = (adv - img).transpose(1, 2,
                                             0)
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = adv[..., ::-1]  # RGB to BGR
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        perturbation = perturbation * 255
        perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)
        fgsmrows = []
        fgsmrows.append([filename[47:75], classes[int(correct)].split(',')[0], (classes[pred].split(',')[0]),
                         (classes[pred2].split(',')[0]), (classes[pred_adv].split(',')[0]),
                         classes[pred_adv2].split(',')[0]])
        with open(fgsmcsv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(fgsmrows)
        counter = counter + 1
    with open(fgsmcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Original Accuracy: " + str(Net1Accuracy / 5000)])
        csvwriter.writerows(["Original Perturbed Accuracy: " + str(Net1FGSMAccuracy / 5000)])
        csvwriter.writerows(["Transfered Accuracy: " + str(Net2Accuracy / 5000)])
        csvwriter.writerows(["Transfered Perturbed Accuracy: " + str(Net2FGSMAccuracy / 5000)])
        csvwriter.writerows(['Original Network: ' + 'resnet34'])
        csvwriter.writerows(['Transfered Network: ' + 'resnet101'])

#Transfer testing for hybrid with CIFAR10 dataset, select first architecture to generate perturbations with, select second architecture to test against, epsilon value, and csv name
def CIFARHybridTesting(original_net, transfer_net, eps, csvname):
    original_net.cuda()
    transfer_net.cuda()
    original_net.eval()
    transfer_net.eval()
    Net1Accuracy = 0
    Net1Hybrid2Accuracy = 0
    Net2Accuracy = 0
    Net2Hybrid2Accuracy = 0
    hybridcsv = csvname
    fieldnames = ['Image', 'Correct Label', 'Network 1 Orig Label', 'Network 2 Orig Label', 'Network 1 Pert Label',
                  'Network 2 Pert Label']
    with open(hybridcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fieldnames)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    transform = transforms.Compose(
        [transforms.ToTensor()
         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)

    for i,data in enumerate(testset):
        print("\n\n\n\n\n\n\n\n\n****************Hybrid Testing *********************\n")
        inputs, labels = data
        filename = i
        if i == 5000:
            break
        start_time = time.time()
        r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool_hybrid2(inputs, original_net, eps)
        correct = labels
        str_label_correct = classes[np.int(correct)]
        str_label_orig = classes[np.int(label_orig)]
        str_label_pert = classes[np.int(label_pert)]
        print("Correct label = ", str_label_correct)
        print("Network 1 Original label = ", str_label_orig)
        print("Network 1 Perturbed label = ", str_label_pert)
        if (int(label_orig) == int(correct)):

            print("Network 1 Classifier is correct on original image")
            Net1Accuracy = Net1Accuracy + 1

        if (int(label_pert) == int(correct)):
            print("Network 1 Classifier is correct on perturbed image")
            Net1Hybrid2Accuracy = Net1Hybrid2Accuracy + 1
        label2 = transfer_net(inputs[None, ...].cuda())
        label2 = np.argmax(label2.detach().cpu().numpy())
        str_label_orig2 = classes[np.int(label2)].split(',')[0]
        label_pert2 = transfer_net(pert_image)
        label_pert2 = np.argmax(label_pert2.detach().cpu().numpy())
        str_label_pert2 = classes[np.int(label_pert2)].split(',')[0]

        print("Network 2 Original label = ", str_label_orig2)
        print("Network 2 Perturbed label = ", str_label_pert2)

        if (int(label2) == int(correct)):
            print("Network 2 Classifier is correct on original image")
            Net2Accuracy = Net2Accuracy + 1

        if (int(label_pert2) == int(correct)):
            print("Network 2 Classifier is correct on perturbed image")
            Net2Hybrid2Accuracy = Net2Hybrid2Accuracy + 1
        hybrows = []
        hybrows.append(
            [filename, str_label_correct, str_label_orig, str_label_orig2, str_label_pert, str_label_pert2])
        with open(hybridcsv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(hybrows)
    with open(hybridcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Original Accuracy: " + str(Net1Accuracy / 5000)])
        csvwriter.writerows(["Original Perturbed Accuracy: " + str(Net1Hybrid2Accuracy / 5000)])
        csvwriter.writerows(["Transfered Accuracy: " + str(Net2Accuracy / 5000)])
        csvwriter.writerows(["Transfered Perturbed Accuracy: " + str(Net2Hybrid2Accuracy / 5000)])
        csvwriter.writerows(['Original Network: ' + 'resnet34'])
        csvwriter.writerows(['Transfered Network: ' + 'resnet101'])


#Define transfer tests to be done

#TransferTestingFGSM(orig_googlenet, eval_resnet101, 0.01, 'transfer_fgsm_resnet34_to_resnet101_0.01.csv')
#TransferTestingDeepfool(orig_googlenet, eval_resnet101, 'transfer_deepfool_resnet34_to_resnet101.csv')
#TransferTestingFGSM(orig_googlenet, eval_resnet101, 0.2, 'transfer_fgsm_resnet34_to_resnet101_0.2.csv')
#TransferTestingFGSM(orig_googlenet, eval_resnet101, 0.05, 'transfer_fgsm_resnet34_to_resnet101_0.05.csv')
#TransferTestingFGSM(orig_googlenet, eval_resnet101, 0.0005, 'transfer_fgsm_resnet34_to_resnet101_0.0005.csv')


#TransferTestingFGSM(orig_alexnet, eval_googlenet, 0.01, 'transfer_fgsm_alexnet_to_googlenet_0.01.csv')
#TransferTestingDeepfool(orig_alexnet, eval_googlenet, 'transfer_deepfool_alexnet_to_googlenet.csv')
#TransferTestingFGSM(orig_alexnet, eval_googlenet, 0.2, 'transfer_fgsm_alexnet_to_googlenet_0.2.csv')
#TransferTestingFGSM(orig_alexnet, eval_googlenet, 0.05, 'transfer_fgsm_alexnet_to_googlenet_0.05.csv')
#TransferTestingFGSM(orig_alexnet, eval_googlenet, 0.0005, 'transfer_fgsm_alexnet_to_googlenet_0.0005.csv')


#TransferTestingFGSM(orig_alexnet, eval_resnet34, 0.01, 'transfer_fgsm_alexnet_to_resnet34_0.01.csv')
#TransferTestingDeepfool(orig_alexnet, eval_resnet34, 'transfer_deepfool_alexnet_to_resnet34.csv')
#TransferTestingFGSM(orig_alexnet, eval_resnet34, 0.2, 'transfer_fgsm_alexnet_to_resnet34_0.2.csv')
#TransferTestingFGSM(orig_alexnet, eval_resnet34, 0.05, 'transfer_fgsm_alexnet_to_resnet34_0.05.csv')
#TransferTestingFGSM(orig_alexnet, eval_resnet34, 0.0005, 'transfer_fgsm_alexnet_to_resnet34_0.0005.csv')

#TransferTestingHybrid(orig_resnet34, eval_resnet101, 0.2, 'transfer_hybrid_resnet34_to_resnet101_0.2.csv')
#TransferTestingHybrid(orig_resnet34, eval_googlenet, 0.2, 'transfer_hybrid_resnet34_to_googlenet_0.2.csv')
#TransferTestingHybrid(orig_resnet34, eval_alexnet, 0.2, 'transfer_hybrid_resnet34_to_alexnet_0.2.csv')
#TransferTestingHybrid(orig_resnet101, eval_resnet34, 0.2, 'transfer_hybrid_resnet101_to_resnet34_0.2.csv')
#TransferTestingHybrid(orig_resnet101, eval_googlenet, 0.2, 'transfer_hybrid_resnet101_to_googlenet_0.2.csv')
#TransferTestingHybrid(orig_alexnet, eval_googlenet, 0.2, 'transfer_hybrid_alexnet_to_googlenet_0.2.csv')
#TransferTestingHybrid(orig_alexnet, eval_resnet34, 0.2, 'transfer_hybrid_alexnet_to_resnet34_0.2.csv')
#TransferTestingHybrid(orig_googlenet, eval_resnet101, 0.2, 'transfer_hybrid_googlenet_to_resnet101_0.2.csv')
#TransferTestingHybrid(orig_googlenet, eval_resnet34, 0.2, 'transfer_hybrid_googlenet_to_resnet34_0.2.csv')
#TransferTestingHybrid(orig_googlenet, eval_alexnet, 0.2, 'transfer_hybrid_googlenet_to_alexnet_0.2.csv')

#TransferTestingFGSM(orig_resnet101, source, 0.01, 'transfer_fgsm_googlenet_to_resnet101_0.01.csv')
#TransferTestingDeepfool(orig_resnet101, source, 'transfer_deepfool_googlenet_to_resnet101.csv')
#TransferTestingHybrid(net, net2, 0.01)
#TransferTestingFGSM(orig_resnet101, source, 0.2, 'transfer_fgsm_googlenet_to_resnet101_0.2.csv')
#TransferTestingFGSM(orig_resnet101, source, 0.05, 'transfer_fgsm_googlenet_to_resnet101_0.05.csv')
#TransferTestingFGSM(orig_resnet101, source, 0.0005, 'transfer_fgsm_googlenet_to_resnet101_0.0005.csv')

CIFARHybridTesting(orig_alexnet, eval_resnet34, 0.0005, 'transfer_hybrid_CIFAR10_alexnet_to_resnet34_0.0005')
CIFARHybridTesting(orig_alexnet, eval_resnet34, 0.05, 'transfer_hybrid_CIFAR10_alexnet_to_resnet34_0.05')
CIFARHybridTesting(orig_alexnet, eval_resnet34, 0.2, 'transfer_hybrid_CIFAR10_alexnet_to_resnet34_0.2')