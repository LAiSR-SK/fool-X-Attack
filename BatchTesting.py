import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
from deepfool_hybrid2 import deepfool_hybrid2
import os
import time
import glob
from imagenet_labels import classes
import cv2
import csv

#Batch testing file for evaluation of hybrid method, deepfool, and FGSM, outputs result csv files. Evaluates each algorithm on 5000 images of ILSVRC validation dataset

#Choose different network architecture to test
#net = models.resnet101(pretrained=True)
net = models.alexnet(pretrained=True)
#net = models.googlenet(pretrained=True)
#net = models.resnet101(pretrained=True)
net.eval()


#Set mean and standard deviation for normalizing image
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

#Define variables for results
deepFool_Testing_Results = ""
hybridApproach_Testing_Results = ""
FGSM_Testing_Results = ""
Accuracy = 0
DeepfoolAccuracy = 0
Hybrid2Accuracy = 0
FGSMAccuracy = 0
DeepfoolAvgFk = 0
HybridAvgFk = 0
FGSMAvgFk = 0
DeepfoolAvgDiff = 0
HybridAvgDiff = 0
FGSMAvgDiff = 0
DeepfoolAvgFroDiff = 0
HybridAvgFroDiff = 0
FGSMAvgFroDiff = 0
deepfoolcsv = 'deepfoolRnet34ILSVRC.csv'
hybridcsv = 'hybridRnet34ILSVRC0005.csv'
fgsmcsv = 'fgsmRnet34ILSVRC0005.csv'
fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Perturbed Label', 'Memory Usage', 'Iterations', 'Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']



dfrows = []
hybrows = []
fgsmrows = []


#Define epsilon value, initialize counter to 0
eps = 0.0005
counter = 0

#Create csvwriter for each csv file
with open(deepfoolcsv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(fieldnames)

with open(hybridcsv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(fieldnames)

with open(fgsmcsv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(fieldnames)


#Open ILSVRC label file
ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')


for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'): #assuming jpg

  print("\n\n\n\n\n\n\n\n\n****************DeepFool Testing *********************\n")
  #Open image
  im_orig=Image.open(filename).convert('RGB')
  print(filename[47:75])
  if counter == 5000:
      break
  #Normalize image, convert to tensor.
  im = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                        std=std)])(im_orig)
  #Start timer
  start_time = time.time()
  #Input values to deepfool
  r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool(im, net)
  print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
  #Stop timer
  end_time = time.time()
  #Calculate execution time
  execution_time = end_time - start_time
  print("execution time = " + str(execution_time))

  #Open synset labels
  labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

  correct = ILSVRClabels[np.int(counter)].split(' ')[1]

  #Get string representations of labels
  str_label_correct = labels[np.int(correct)].split(',')[0]
  str_label_orig = labels[np.int(label_orig)].split(',')[0]
  str_label_pert = labels[np.int(label_pert)].split(',')[0]


  print("Correct label = ", str_label_correct)
  print("Original label = ", str_label_orig)
  print("Perturbed label = ", str_label_pert)
  #If original label from classifier matches correct, add to accuracy count.
  if(int(label_orig) == int(correct)):
      print("Classifier is correct")
      Accuracy = Accuracy+1
  #If label from perturbed image matches correct, add to accuracy count.
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
  #Calculate difference between perturbed image and original image
  diff = imagetransform(pert_image.cpu()[0]) - tensortransform(im_orig)
  #Calculate frobenius of difference
  fro = np.linalg.norm(diff.numpy())
  #Calculate average distance
  average = torch.mean(torch.abs(diff))
  #Add to average values
  DeepfoolAvgFk = DeepfoolAvgFk + newf_k
  DeepfoolAvgDiff = DeepfoolAvgDiff + average
  DeepfoolAvgFroDiff = DeepfoolAvgFroDiff + fro
  dfrows = []
  #Append values to rows, append to csv file
  dfrows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert, torch.cuda.memory_stats('cuda:0')['active.all.current'], str(loop_i), str(execution_time), newf_k, average, fro])
  with open(deepfoolcsv, 'a', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)

      csvwriter.writerows(dfrows)

  print("##################################  END DEEP FOOL TESTING ##############################################################\n")






  print(" \n\n\n**************** Hybrid Approach DeepFool 2 *********************\n" )
  im_orig=Image.open(filename).convert('RGB')
  print(filename)
  im = transforms.Compose([
      transforms.Scale(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean,
                           std=std)])(im_orig)
  start_time = time.time()
  #Input values to hybrid method
  r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool_hybrid2(im, net, eps)
  print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
  end_time = time.time()
  execution_time = end_time - start_time
  print("execution time = " + str(execution_time))

  labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

  str_label_orig = labels[np.int(label_orig)].split(',')[0]
  str_label_pert = labels[np.int(label_pert)].split(',')[0]


  print("Correct label = ", str_label_correct)
  print("Original label = ", str_label_orig)
  print("Perturbed label = ", str_label_pert)

  #If perturbed label matches correct label, add to accuracy count
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

  print("Iterations: " + str(loop_i))
  diff = imagetransform(pert_image.cpu()[0]) - tensortransform(im_orig)
  fro = np.linalg.norm(diff.numpy())
  average = torch.mean(torch.abs(diff))
  HybridAvgFk = HybridAvgFk + newf_k
  HybridAvgDiff = HybridAvgDiff + average
  HybridAvgFroDiff = HybridAvgFroDiff + fro
  hybrows = []
  hybrows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert, torch.cuda.memory_stats('cuda:0')['active.all.current'], str(loop_i), str(execution_time), newf_k, average, fro])
  with open(hybridcsv, 'a', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)

      csvwriter.writerows(hybrows)

  print("#################################### END Hybrid Testing ############################################################\n")



  print(" **************** FGSM Testing *********************\n")



  print("FGSM")
  #Open image
  orig = cv2.imread(filename)[..., ::-1]
  print(filename)
  start_time = time.time()
  orig = cv2.resize(orig, (224, 224))
  img = orig.copy().astype(np.float32)
  #create empty array for perturbation
  perturbation = np.empty_like(orig)

  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  img /= 255.0
  img = (img - mean) / std
  img = img.transpose(2, 0, 1)
  #Convert image into pytorch tensor, input image into network
  inp = Variable(torch.from_numpy(img).to('cuda:0').float().unsqueeze(0), requires_grad=True)

  out = net(inp)
  criterion = nn.CrossEntropyLoss()
  pred = np.argmax(out.data.cpu().numpy())
  loss = criterion(out, Variable(torch.Tensor([float(pred)]).to('cuda:0').long()))
  print('Prediction before attack: %s' % (classes[pred].split(',')[0]))

  # compute gradients
  loss.backward()
  grad_orig = inp.grad.data.cpu().numpy().copy()

  # this is it, this is the method
  inp.data = inp.data + (eps * torch.sign(inp.grad.data))
  print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
  inp.grad.data.zero_()  # unnecessary

  end_time = time.time()
  execution_time = end_time - start_time
  print("execution time = " + str(execution_time))

  # predict on the adversarial image
  pred_adv = np.argmax(net(inp).data.cpu().numpy())
  print("After attack: " + str(eps) + " " + classes[pred_adv].split(',')[0])
  grad_cur = inp.grad.data.cpu().numpy().copy()
  w_k = grad_cur - grad_orig
  fs = net.forward(inp)
  f_k = (fs[0, pred_adv] - fs[0, int(correct)]).data.cpu().numpy()
  if(int(pred_adv) == int(correct)):
      print("Classifier is correct")
      FGSMAccuracy = FGSMAccuracy + 1

  # deprocess image
  adv = inp.data.cpu().numpy()[0]
  perturbation = (adv - img).transpose(1, 2,
                                       0)
  adv = adv.transpose(1, 2, 0)
  adv = (adv * std) + mean
  adv = adv * 255.0
  adv = adv[..., ::-1]  # RGB to BGR
  adv = np.clip(adv, 0, 255).astype(np.uint8)
  diff = imagetransform(inp.data.cpu()[0]) - tensortransform(im_orig)
  fro = np.linalg.norm(diff.numpy())
  average = torch.mean(torch.abs(diff))
  perturbation = perturbation * 255
  perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)
  FGSMAvgFk = FGSMAvgFk + f_k
  FGSMAvgDiff = FGSMAvgDiff + average
  FGSMAvgFroDiff = FGSMAvgFroDiff + fro
  fgsmrows = []
  fgsmrows.append([filename[47:75], classes[int(correct)].split(',')[0], (classes[pred].split(',')[0]), (classes[pred_adv].split(',')[0]), torch.cuda.memory_stats('cuda:0')['active.all.current'], str(loop_i), str(execution_time), f_k, average, fro])
  with open(fgsmcsv, 'a', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)

      csvwriter.writerows(fgsmrows)
  print("######################################## END FGSM TESTING ########################################################\n")
  counter = counter+1

#Add total values to csv file
with open(deepfoolcsv, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(["Epsilon: " + str(eps)])
    csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
    csvwriter.writerows(["Perturbed Accuracy: " + str(DeepfoolAccuracy/5000)])
    csvwriter.writerows(["Avg F_k: " + str(DeepfoolAvgFk/5000)])
    csvwriter.writerows(["Avg Difference: " + str(DeepfoolAvgDiff / 5000)])
    csvwriter.writerows(["Avg Frobenius of Difference: " + str(DeepfoolAvgFroDiff / 5000)])
with open(hybridcsv, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(["Epsilon: " + str(eps)])
    csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
    csvwriter.writerows(["Perturbed Accuracy: " + str(Hybrid2Accuracy/5000)])
    csvwriter.writerows(["Avg F_k: " + str(HybridAvgFk/5000)])
    csvwriter.writerows(["Avg Difference: " + str(HybridAvgDiff / 5000)])
    csvwriter.writerows(["Avg Frobenius of Difference: " + str(HybridAvgFroDiff/5000)])
with open(fgsmcsv, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(["Epsilon: " + str(eps)])
    csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
    csvwriter.writerows(["Perturbed Accuracy: " + str(FGSMAccuracy/5000)])
    csvwriter.writerows(["Avg F_k: " + str(FGSMAvgFk/5000)])
    csvwriter.writerows(["Avg Difference: " + str(FGSMAvgDiff / 5000)])
    csvwriter.writerows(["Avg Frobenius of Difference: " + str(FGSMAvgFroDiff/5000)])