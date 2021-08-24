import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from deepfool_hybrid2 import deepfool_hybrid2
import os
import time
import glob
import csv

#Testing hybrid approach with L Inf and L 0 Norm, must be set in deepfool_hybrid2

eps = 0.005

net = models.alexnet(pretrained=True)
net.eval()

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

hybrows = []
counter = 0
hybridApproach_Testing_Results = ""
Accuracy = 0
Hybrid2Accuracy = 0
HybridAvgTime = 0
HybridAvgFk = 0
HybridAvgDiff = 0
HybridAvgFroDiff = 0
hybridcsv = 'hybridalexnetlinf.csv'
fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Perturbed Label', 'Memory Usage',
                  'Iterations', 'Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']

with open(hybridcsv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(fieldnames)


ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'): #assuming jpg
    im_orig = Image.open(filename).convert('RGB')
    print(filename[47:75])
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
    start_time = time.time()
    r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool_hybrid2(im, net, eps)
    print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
    end_time = time.time()
    execution_time = end_time - start_time
    print("execution time = " + str(execution_time))
    HybridAvgTime = HybridAvgTime + execution_time

    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    correct = ILSVRClabels[np.int(counter)].split(' ')[1]

    str_label_correct = labels[np.int(correct)].split(',')[0]
    str_label_orig = labels[np.int(label_orig)].split(',')[0]
    str_label_pert = labels[np.int(label_pert)].split(',')[0]

    print("Correct label = ", str_label_correct)
    print("Original label = ", str_label_orig)
    print("Perturbed label = ", str_label_pert)

    if (int(label_orig) == int(correct)):
        print("Classifier is correct")
        Accuracy = Accuracy + 1

    if (int(label_pert) == int(correct)):
        print("Classifier is correct")
        Hybrid2Accuracy = Hybrid2Accuracy + 1


    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv * torch.ones(A.shape))
        A = torch.min(A, maxv * torch.ones(A.shape))
        return A


    clip = lambda x: clip_tensor(x, 0, 255)

    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                             transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                             transforms.Lambda(clip),
                             transforms.ToPILImage(),
                             transforms.CenterCrop(224)])

    imagetransform = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                         transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                         transforms.Lambda(clip)])

    tensortransform = transforms.Compose([transforms.Scale(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                          transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                          transforms.Lambda(clip)])
    print("Iterations: " + str(loop_i))
    diff = imagetransform(pert_image.cpu()[0]) - tensortransform(im_orig)
    fro = np.linalg.norm(diff.numpy())
    average = torch.mean(torch.abs(diff))
    HybridAvgFk = HybridAvgFk + newf_k
    HybridAvgDiff = HybridAvgDiff + average
    HybridAvgFroDiff = HybridAvgFroDiff + fro
    hybrows = []
    hybrows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert,
                    torch.cuda.memory_stats('cuda:0')['active.all.current'], str(loop_i), str(execution_time), newf_k,
                    average, fro])
    with open(hybridcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerows(hybrows)

    print("#################################### END Hybrid Testing ############################################################\n")
    counter = counter + 1

with open(hybridcsv, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(["Epsilon: " + str(eps)])
    csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
    csvwriter.writerows(["Perturbed Accuracy: " + str(Hybrid2Accuracy/5000)])
    csvwriter.writerows(["Avg F_k: " + str(HybridAvgFk/5000)])
    csvwriter.writerows(["Avg Difference: " + str(HybridAvgDiff / 5000)])
    csvwriter.writerows(["Avg Frobenius of Difference: " + str(HybridAvgFroDiff/5000)])
    csvwriter.writerows(["Avg Time: " + str(HybridAvgTime / 5000)])