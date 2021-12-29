import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models import AlexNet
import torch
import torch.utils.data as data_utils
from torch.autograd import Variable
from deepfool import deepfool
from foolx import foolx
import time
from imagenet_labels import classes
import csv

#Basic testing for evaluating foolx, deepfool, FGSM on CIFAR10


#Models compatible with CIFAR10 provided by: https://github.com/icpm/pytorch-cifar10
#net = models.resnet34(pretrained=False)
#net = models.alexnet(pretrained=False)
net = AlexNet.AlexNet()
#net = LeNet.LeNet()
state = torch.load("models/alexnet/model.pth")
net.load_state_dict(state)
net.eval()



mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]


deepFool_Testing_Results = ""
foolXApproach_Testing_Results = ""
FGSM_Testing_Results = ""
Accuracy = 0
DeepfoolAccuracy = 0
FoolXAccuracy = 0
FGSMAccuracy = 0
DeepfoolAvgFk = 0
FoolXAvgFk = 0
FGSMAvgFk = 0
DeepfoolAvgDiff = 0
FoolXAvgDiff = 0
FGSMAvgDiff = 0
DeepfoolAvgFroDiff = 0
FoolXAvgFroDiff = 0
FGSMAvgFroDiff = 0
deepfoolcsv = 'deepfoolAlexNetCIFAR10.csv'
foolxcsv = 'foolxAlexNetCIFAR10.csv'
fgsmcsv = 'fgsmAlexNetCIFAR10.csv'
fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Perturbed Label', 'Memory Usage', 'Iterations', 'Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']

#Check if cuda is available.
is_cuda = torch.cuda.is_available()
device = 'cpu'

#If cuda is available use GPU for faster processing, if not, use CPU.
if is_cuda:
    print("Using GPU")
    device = 'cuda:0'
else:
    print("Using CPU")

dfrows = []
hybrows = []
fgsmrows = []


deepfoolsamples = []
foolxsamples = []
fgsmsamples = []


with open(deepfoolcsv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(fieldnames)

with open(foolxcsv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(fieldnames)

with open(fgsmcsv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(fieldnames)

#Define classes for CIFAR10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Transform to convert images to tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
     ])

batch_size = 4

#Define train and test sets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

def main():
    counter = 0
    global Accuracy
    global DeepfoolAccuracy
    global FoolXAccuracy
    global FGSMAccuracy
    global DeepfoolAvgFk
    global FoolXAvgFk
    global FGSMAvgFk
    global DeepfoolAvgDiff
    global FoolXAvgDiff
    global FGSMAvgDiff
    global DeepfoolAvgFroDiff
    global FoolXAvgFroDiff
    global FGSMAvgFroDiff
    for i,data in enumerate(testset):
      # print("\n################################################################################################\n")
      eps = 0.05
      print("\n\n\n\n\n\n\n\n\n****************DeepFool Testing *********************\n")
      inputs, labels = data
      filename = i
      if i == 5000:
          break
      start_time = time.time()
      r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool(inputs, net)
      print("Memory Usage: ", torch.cuda.memory_stats(device)['active.all.current'])
      end_time = time.time()
      execution_time = end_time - start_time
      print("execution time = " + str(execution_time))

      correct = labels

      str_label_correct = classes[np.int(correct)]
      str_label_orig = classes[np.int(label_orig)]
      str_label_pert = classes[np.int(label_pert)]


      print("Correct label = ", str_label_correct)
      print("Original label = ", str_label_orig)
      print("Perturbed label = ", str_label_pert)
      if(int(label_orig) == int(correct)):
        print("Classifier is correct")
        Accuracy = Accuracy+1

      if (int(label_pert) == int(correct)):
        print("Classifier is correct")
        DeepfoolAccuracy = DeepfoolAccuracy + 1


      def clip_tensor(A, minv, maxv):
           A = torch.max(A, minv * torch.ones(A.shape))
           A = torch.min(A, maxv * torch.ones(A.shape))
           return A

      clip = lambda x: clip_tensor(x, 0, 255)

      imagetransform = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                            transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                            transforms.Lambda(clip)])
      print("Iterations: " + str(loop_i))
      diff = imagetransform(pert_image.cpu()[0]) - inputs
      fro = np.linalg.norm(diff.numpy())
      average = torch.mean(torch.abs(diff))
      DeepfoolAvgFk = DeepfoolAvgFk + newf_k
      DeepfoolAvgDiff = DeepfoolAvgDiff + average
      DeepfoolAvgFroDiff = DeepfoolAvgFroDiff + fro
      dfrows = []
      dfrows.append([filename, str_label_correct, str_label_orig, str_label_pert, torch.cuda.memory_stats(device)['active.all.current'], str(loop_i), str(execution_time), newf_k, average, fro])
      with open(deepfoolcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerows(dfrows)

      print("##################################  END DEEP FOOL TESTING ##############################################################\n")






      print(" \n\n\n**************** Fool-X *********************\n" )
      print (filename)
      start_time = time.time()
      r, loop_i, label_orig, label_pert, pert_image, newf_k = foolx(inputs, net, eps)
      print("Memory Usage: ", torch.cuda.memory_stats(device)['active.all.current'])
      end_time = time.time()
      execution_time = end_time - start_time
      print("execution time = " + str(execution_time))

      correct = labels

      str_label_correct = classes[np.int(correct)]
      str_label_orig = classes[np.int(label_orig)]
      str_label_pert = classes[np.int(label_pert)]


      print("Correct label = ", str_label_correct)
      print("Original label = ", str_label_orig)
      print("Perturbed label = ", str_label_pert)


      if (int(label_pert) == int(correct)):
          print("Classifier is correct")
          FoolXAccuracy = FoolXAccuracy + 1

      if (int(label_orig) == int(correct)):
            print("Classifier is correct")
            Accuracy = Accuracy+1

      print("Iterations: " + str(loop_i))
      diff = imagetransform(pert_image.cpu()[0]) - inputs
      fro = np.linalg.norm(diff.numpy())
      average = torch.mean(torch.abs(diff))
      FoolXAvgFk = FoolXAvgFk + newf_k
      FooolXAvgDiff = FoolXAvgDiff + average
      FoolXAvgFroDiff = FoolXAvgFroDiff + fro
      hybrows = []
      hybrows.append([filename, str_label_correct, str_label_orig, str_label_pert, torch.cuda.memory_stats(device)['active.all.current'], str(loop_i), str(execution_time), newf_k, average, fro])
      with open(foolxcsv, 'a', newline='') as csvfile:
          csvwriter = csv.writer(csvfile)

          csvwriter.writerows(hybrows)

      print("#################################### END Fool-X Testing ############################################################\n")



      print(" **************** FGSM Testing *********************\n")



      print("FGSM")
      print(filename)
      start_time = time.time()


      inp = Variable(torch.from_numpy(inputs.numpy()).to(device).float().unsqueeze(0), requires_grad=True)

      out = net(inp)
      criterion = nn.CrossEntropyLoss()
      pred = np.argmax(out.data.cpu().numpy())
      loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))
      print('Prediction before attack: %s' % (classes[pred]))

      # compute gradients
      loss.backward()

      # this is it, this is the method
      inp.data = inp.data + (eps * torch.sign(inp.grad.data))
      print("Memory Usage: ", torch.cuda.memory_stats(device)['active.all.current'])
      inp.grad.data.zero_()  # unnecessary

      end_time = time.time()
      execution_time = end_time - start_time
      print("execution time = " + str(execution_time))

      # predict on the adversarial image
      pred_adv = np.argmax(net(inp).data.cpu().numpy())
      print("After attack: " + str(eps) + " " + classes[pred_adv])
      fs = net.forward(inp)
      f_k = (fs[0, pred_adv] - fs[0, int(correct)]).data.cpu().numpy()
      if(int(pred_adv) == int(correct)):
        print("Classifier is correct")
        FGSMAccuracy = FGSMAccuracy + 1
      diff = imagetransform(inp.data.cpu()[0]) - inputs
      fro = np.linalg.norm(diff.numpy())
      average = torch.mean(torch.abs(diff))
      FGSMAvgFk = FGSMAvgFk + f_k
      FGSMAvgDiff = FGSMAvgDiff + average
      FGSMAvgFroDiff = FGSMAvgFroDiff + fro
      fgsmrows = []
      fgsmrows.append([filename, classes[int(correct)], (classes[pred]), (classes[pred_adv]), torch.cuda.memory_stats(device)['active.all.current'], str(loop_i), str(execution_time), f_k, average, fro])
      with open(fgsmcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerows(fgsmrows)
      print("######################################## END FGSM TESTING ########################################################\n")
      counter = counter+1

    with open(deepfoolcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(DeepfoolAccuracy/5000)])
        csvwriter.writerows(["Avg F_k: " + str(DeepfoolAvgFk/5000)])
        csvwriter.writerows(["Avg Difference: " + str(DeepfoolAvgDiff / 5000)])
        csvwriter.writerows(["Avg Frobenius of Difference: " + str(DeepfoolAvgFroDiff / 5000)])
    with open(foolxcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(FoolXAccuracy/5000)])
        csvwriter.writerows(["Avg F_k: " + str(FoolXAvgFk/5000)])
        csvwriter.writerows(["Avg Difference: " + str(FoolXAvgDiff / 5000)])
        csvwriter.writerows(["Avg Frobenius of Difference: " + str(FoolXAvgFroDiff/5000)])
    with open(fgsmcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(FGSMAccuracy/5000)])
        csvwriter.writerows(["Avg F_k: " + str(FGSMAvgFk/5000)])
        csvwriter.writerows(["Avg Difference: " + str(FGSMAvgDiff / 5000)])
        csvwriter.writerows(["Avg Frobenius of Difference: " + str(FGSMAvgFroDiff/5000)])


if __name__ == '__main__':
    main()
