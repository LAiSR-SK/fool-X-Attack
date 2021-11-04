import torchvision.transforms as transforms
from deepfool import deepfool
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from foolx import foolx
import os
import time
import glob
import csv

eps = 0.005

net = models.googlenet(pretrained=True)
net.eval()

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

hybrows = []
dfrows = []
counter = 0
hybridApproach_Testing_Results = ""
Accuracy = 0
DeepfoolAccuracy = 0
DeepfoolAvgTime = 0
DeepfoolAvgFk = 0
DeepfoolAvgDiff = 0
DeepfoolAvgFroDiff = 0
Hybrid2Accuracy = 0
HybridAvgTime = 0
HybridAvgFk = 0
HybridAvgDiff = 0
HybridAvgFroDiff = 0
hybridcsv = 'hybridgooglenetl1.csv'
deepfoolcsv = 'deepfoolgooglenetl1.csv'
fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Perturbed Label', 'Memory Usage',
                  'Iterations', 'Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']


ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

for filename in glob.glob('D:/ImageNet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'): #assuming jpg
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
    r, loop_i, label_orig, label_pert, pert_image, newf_k = foolx(im, net, eps)
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

    print("\n\n\n\n\n\n\n\n\n****************DeepFool Testing *********************\n")
    # Open image
    im_orig = Image.open(filename).convert('RGB')
    print(filename[47:75])
    if counter == 5000:
        break
    # Normalize image, convert to tensor.
    im = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)])(im_orig)
    # Start timer
    start_time = time.time()
    # Input values to deepfool
    r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool(im, net)
    print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
    # Stop timer
    end_time = time.time()
    # Calculate execution time
    execution_time = end_time - start_time
    print("execution time = " + str(execution_time))

    # Open synset labels
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    correct = ILSVRClabels[np.int(counter)].split(' ')[1]

    # Get string representations of labels
    str_label_correct = labels[np.int(correct)].split(',')[0]
    str_label_orig = labels[np.int(label_orig)].split(',')[0]
    str_label_pert = labels[np.int(label_pert)].split(',')[0]

    print("Correct label = ", str_label_correct)
    print("Original label = ", str_label_orig)
    print("Perturbed label = ", str_label_pert)
    # If original label from classifier matches correct, add to accuracy count.
    if (int(label_orig) == int(correct)):
        print("Classifier is correct")
        Accuracy = Accuracy + 1
    # If label from perturbed image matches correct, add to accuracy count.
    if (int(label_pert) == int(correct)):
        print("Classifier is correct")
        DeepfoolAccuracy = DeepfoolAccuracy + 1


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
    # Calculate difference between perturbed image and original image
    diff = imagetransform(pert_image.cpu()[0]) - tensortransform(im_orig)
    # Calculate frobenius of difference
    fro = np.linalg.norm(diff.numpy())
    # Calculate average distance
    average = torch.mean(torch.abs(diff))
    # Add to average values
    DeepfoolAvgFk = DeepfoolAvgFk + newf_k
    DeepfoolAvgDiff = DeepfoolAvgDiff + average
    DeepfoolAvgFroDiff = DeepfoolAvgFroDiff + fro
    DeepfoolAvgTime = DeepfoolAvgTime + execution_time
    dfrows = []
    # Append values to rows, append to csv file
    dfrows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert,
                   torch.cuda.memory_stats('cuda:0')['active.all.current'], str(loop_i), str(execution_time), newf_k,
                   average, fro])
    with open(deepfoolcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerows(dfrows)

    print("##################################  END DEEP FOOL TESTING ##############################################################\n")

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
with open(deepfoolcsv, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(["Epsilon: " + str(eps)])
    csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
    csvwriter.writerows(["Perturbed Accuracy: " + str(DeepfoolAccuracy/5000)])
    csvwriter.writerows(["Avg F_k: " + str(DeepfoolAvgFk/5000)])
    csvwriter.writerows(["Avg Difference: " + str(DeepfoolAvgDiff / 5000)])
    csvwriter.writerows(["Avg Frobenius of Difference: " + str(DeepfoolAvgFroDiff / 5000)])
    csvwriter.writerows(["Avg Time: " + str(DeepfoolAvgTime / 5000)])