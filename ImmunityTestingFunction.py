import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from deepfool import deepfool
from foolx import foolx
import os
import time
import glob
from imagenet_labels import classes
import cv2
import csv

#Check if cuda is available.
is_cuda = torch.cuda.is_available()
device = 'cpu'

#If cuda is available use GPU for faster processing, if not, use CPU.
if is_cuda:
    print("Using GPU")
    device = 'cuda:0'
else:
    print("Using CPU")

#Contains functions called for testing immunity of deepfool, foolx, and FGSM.

#testing function for all methods, takes an original network and 3 networks finetuned on each method as input
def testingFunction(net_orig, net_df, net_hyb, net_fgsm):
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]


    Accuracy = 0
    DeepfoolAccuracy = 0
    foolXAccuracy = 0
    FGSMAccuracy = 0
    deepfoolcsv = 'deepfoolRnet34ILSVRC.csv'
    foolXcsv = 'foolXRnet34ILSVRC.csv'
    fgsmcsv = 'fgsmRnet34ILSVRC.csv'
    fieldnames = ['Image', 'Correct Label', 'Classified Label Before Perturbation', 'Perturbed Label']


    eps = 0
    counter = 0


    with open(deepfoolcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    with open(foolXcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    with open(fgsmcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)


    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')


    for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'): #assuming jpg
      if int(filename[66:70]) < 5000:
          continue

      print("\n\n\n\n\n\n\n\n\n****************DeepFool Testing *********************\n")
      im_orig=Image.open(filename).convert('RGB')
      print(filename[47:75])
      if counter == 5000:
          break
      im = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std)])(im_orig)
      start_time = time.time()
      r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool(im, net_df)
      print("Memory Usage: ", torch.cuda.memory_stats(device)['active.all.current'])
      end_time = time.time()
      execution_time = end_time - start_time
      print("execution time = " + str(execution_time))

      labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
      correct = ILSVRClabels[np.int(counter)].split(' ')[1]


      str_label_correct = labels[np.int(correct)].split(',')[0]
      str_label_orig = labels[np.int(label_orig)].split(',')[0]
      str_label_pert = labels[np.int(label_pert)].split(',')[0]


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
      dfrows = []
      dfrows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert])
      with open(deepfoolcsv, 'a', newline='') as csvfile:
          csvwriter = csv.writer(csvfile)
          csvwriter.writerows(dfrows)

      print("##################################  END DEEP FOOL TESTING ##############################################################\n")






      print(" \n\n\n**************** Fool-X *********************\n" )
      im_orig=Image.open(filename).convert('RGB')
      print (filename)
      im = transforms.Compose([
          transforms.Scale(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=mean,
                               std=std)])(im_orig)
      start_time = time.time()
      r, loop_i, label_orig, label_pert, pert_image, newf_k = foolx(im, net_hyb, eps)
      print("Memory Usage: ", torch.cuda.memory_stats(device)['active.all.current'])
      end_time = time.time()
      execution_time = end_time - start_time
      print("execution time = " + str(execution_time))

      labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

      str_label_orig = labels[np.int(label_orig)].split(',')[0]
      str_label_pert = labels[np.int(label_pert)].split(',')[0]


      print("Correct label = ", str_label_correct)
      print("Original label = ", str_label_orig)
      print("Perturbed label = ", str_label_pert)


      if (int(label_pert) == int(correct)):
          print("Classifier is correct")
          foolXAccuracy = foolXAccuracy + 1

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
      hybrows = []
      hybrows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert])
      with open(foolXcsv, 'a', newline='') as csvfile:
          csvwriter = csv.writer(csvfile)
          csvwriter.writerows(hybrows)

      print("#################################### END Fool-X Testing ############################################################\n")



      print(" **************** FGSM Testing *********************\n")



      print("FGSM")
      window_adv = 'perturbation'
      orig = cv2.imread(filename)[..., ::-1]
      print(filename)
      start_time = time.time()
      orig = cv2.resize(orig, (224, 224))
      img = orig.copy().astype(np.float32)

      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]
      img /= 255.0
      img = (img - mean) / std
      img = img.transpose(2, 0, 1)

      inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)

      out = net_orig(inp)
      criterion = nn.CrossEntropyLoss()
      pred = np.argmax(out.data.cpu().numpy())
      loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))
      print('Prediction before attack: %s' % (classes[pred].split(',')[0]))

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
      pred_adv = np.argmax(net_fgsm(inp).data.cpu().numpy())
      print("After attack: " + str(eps) + " " + classes[pred_adv].split(',')[0])
      if(int(pred_adv) == int(correct)):
          print("Classifier is correct")
          FGSMAccuracy = FGSMAccuracy + 1

      fgsmrows = []
      fgsmrows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert])
      with open(fgsmcsv, 'a', newline='') as csvfile:
          csvwriter = csv.writer(csvfile)

          csvwriter.writerows(fgsmrows)
      print("######################################## END FGSM TESTING ########################################################\n")
      counter = counter+1


    with open(deepfoolcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(DeepfoolAccuracy/5000)])
        csvwriter.writerows(["Network: ResNet101"])
    with open(foolXcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(foolXAccuracy/5000)])
        csvwriter.writerows(["Network: ResNet101"])
    with open(fgsmcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(FGSMAccuracy/5000)])
        csvwriter.writerows(["Network: ResNet101"])

#Testing function for foolx approach, takes an original network, a network finetuned on images generated by the foolx approach, an epsilon value, and a file name as input
def foolxImmunityTesting(orig_net, foolx_net, eps, csvfilename):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    Accuracy = 0
    foolXAccuracy = 0
    foolXImmunity = 0
    foolXcsv = csvfilename
    fieldnames = ['Image', 'Correct Label', 'Classified Label Before Perturbation', 'Perturbed Label', 'Label from Immune Network']

    with open(foolXcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

    counter = 4999
    for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
        if int(filename[65:70]) < 5000:
            continue
        im_orig = Image.open(filename).convert('RGB')
        print(filename[47:75])
        if counter == 10000:
            break
        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])(im_orig)

        r, loop_i, label_orig, label_pert, pert_image, newf_k = foolx(im, orig_net, eps)


        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]

        str_label_orig_correct = labels[np.int(correct)].split(',')[0]
        str_label_orig_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_orig_pert = labels[np.int(label_pert)].split(',')[0]

        print("Correct label = ", str_label_orig_correct)
        print("Original label (Original Network) = ", str_label_orig_orig)
        print("Perturbed label (Original Network) = ", str_label_orig_pert)

        if is_cuda:
            resultset = foolx_net(im[None, :].cuda())
        else:
            resultset = foolx_net(im[None, :].cpu())
        result = np.argmax(resultset.detach().cpu().numpy())
        str_label_result = labels[np.int(result)].split(',')[0]
        print("Result from Immune Network = ", str_label_result)

        if (int(label_orig) == int(correct)):
            print("Original Classifier is correct")
            Accuracy = Accuracy + 1

        if (int(result) == int(correct)):
            print("Immune Classifier is correct")
            foolXAccuracy = foolXAccuracy + 1

        if (int(result) == int(label_orig)):
            print("Immune Classifier equals original classification")
            foolXImmunity = foolXImmunity + 1

        hybrows = []
        hybrows.append([filename[47:75], str_label_orig_correct, str_label_orig_orig, str_label_orig_pert, str_label_result])
        with open(foolXcsv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(hybrows)
        counter = counter + 1

    with open(foolXcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Original Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(foolXAccuracy/5000)])
        csvwriter.writerows(["Robustness: " + str(foolXImmunity / 5000)])
        csvwriter.writerows(["Network: AlexNet"])

#Testing function for deepfool approach, takes an original network, a network finetuned on images generated by deepfool, and a file name as input
def deepfoolImmunityTesting(orig_net, deepfool_net, csvfilename):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    Accuracy = 0
    DeepfoolAccuracy = 0
    DeepfoolImmunity = 0
    deepfoolcsv = csvfilename
    fieldnames = ['Image', 'Correct Label', 'Classified Label Before Perturbation', 'Perturbed Label',
                  'Label from Immune Network']

    with open(deepfoolcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

    counter = 4999
    for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
        if int(filename[65:70]) < 5000:
            continue
        im_orig = Image.open(filename).convert('RGB')
        print(filename[47:75])
        if counter == 10000:
            break
        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])(im_orig)

        r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool(im, orig_net)

        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]

        str_label_orig_correct = labels[np.int(correct)].split(',')[0]
        str_label_orig_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_orig_pert = labels[np.int(label_pert)].split(',')[0]

        print("Correct label = ", str_label_orig_correct)
        print("Original label (Original Network) = ", str_label_orig_orig)
        print("Perturbed label (Original Network) = ", str_label_orig_pert)

        if is_cuda:
            resultset = deepfool_net(im[None, :].cuda())
        else:
            resultset = deepfool_net(im[None, :].cpu())

        result = np.argmax(resultset.detach().cpu().numpy())
        str_label_result = labels[np.int(result)].split(',')[0]
        print("Result from Immune Network = ", str_label_result)

        if (int(label_orig) == int(correct)):
            print("Original Classifier is correct")
            Accuracy = Accuracy + 1

        if (int(result) == int(correct)):
            print("Immune Classifier is correct")
            DeepfoolAccuracy = DeepfoolAccuracy + 1

        if (int(result) == int(label_orig)):
            print("Immune Classifier equals original classification")
            DeepfoolImmunity = DeepfoolImmunity + 1



        print("Iterations: " + str(loop_i))
        dfrows = []
        dfrows.append([filename[47:75], str_label_orig_correct, str_label_orig_orig, str_label_orig_pert, str_label_result])
        with open(deepfoolcsv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(dfrows)
        counter = counter + 1

    with open(deepfoolcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Original Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(DeepfoolAccuracy / 5000)])
        csvwriter.writerows(["Robustness: " + str(DeepfoolImmunity / 5000)])
        csvwriter.writerows(["Network: AlexNet"])

#Testing function for FGSM approach, takes an original network, a network finetuned on images generated by FGSM, an epsilon value, and a file name as input
def FGSMImmunityTesting(orig_net, fgsm_net, eps, csvfilename):

    Accuracy = 0
    FGSMAccuracy = 0
    FGSMImmunity = 0
    fgsmcsv = csvfilename
    fieldnames = ['Image', 'Correct Label', 'Classified Label Before Perturbation', 'Perturbed Label',
                  'Label from Immune Network']

    with open(fgsmcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

    counter = 4999
    for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
        if int(filename[65:70]) < 5000:
            continue
        print("FGSM")
        if counter == 10000:
            break
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]
        orig = cv2.imread(filename)[..., ::-1]
        print(filename)
        orig = cv2.resize(orig, (224, 224))
        img = orig.copy().astype(np.float32)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img /= 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        eps = eps

        inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)

        out = orig_net(inp)
        criterion = nn.CrossEntropyLoss()
        pred = np.argmax(out.data.cpu().numpy())
        loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))
        print('Prediction before attack: %s' % (classes[pred].split(',')[0]))
        if (int(pred) == int(correct)):
            print("Original Classifier is correct")
            Accuracy = Accuracy + 1

        # compute gradients
        loss.backward()

        # this is it, this is the method
        inp.data = inp.data + (eps * torch.sign(inp.grad.data))
        inp.grad.data.zero_()  # unnecessary


        # predict on the adversarial image
        pred_adv = np.argmax(orig_net(inp).data.cpu().numpy())
        print("Perturbed label: " + str(eps) + " " + classes[pred_adv].split(',')[0])
        pred_adv_immune = np.argmax(fgsm_net(inp).data.cpu().numpy())
        print("Immune perturbed label: " + str(eps) + " " + classes[pred_adv_immune].split(',')[0])
        print(filename[47:75])
        if (int(pred_adv_immune) == int(correct)):
            print("Immune Classifier is correct")
            FGSMAccuracy = FGSMAccuracy + 1

        if (int(pred) == int(pred_adv_immune)):
            print("Immune Classifier equals original classification")
            FGSMImmunity = FGSMImmunity + 1


        fgsmrows = []
        fgsmrows.append([filename[47:75], classes[int(correct)].split(',')[0], (classes[pred].split(',')[0]), (classes[pred_adv].split(',')[0]), (classes[pred_adv_immune].split(',')[0])])
        with open(fgsmcsv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(fgsmrows)
        counter = counter + 1
    with open(fgsmcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Original Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(FGSMAccuracy / 5000)])
        csvwriter.writerows(["Robustness: " + str(FGSMImmunity / 5000)])
        csvwriter.writerows(["Network: AlexNet"])

