import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from foolx import foolx
from deepfool import deepfool
import os
import time
import glob
import csv
import art

#Batch testing file for evaluation of foolx method, outputs result csv files. Evaluates algorithm on 5000 images of ILSVRC validation dataset

#Choose different network architecture to test
netr101 = models.resnet101(pretrained=True)
neta = models.alexnet(pretrained=True)
netg = models.googlenet(pretrained=True)
netr34 = models.resnet34(pretrained=True)
netr101.eval()
neta.eval()
netg.eval()
netr34.eval()
netr101.cuda()
neta.cuda()
netg.cuda()
netr34.cuda()


#Set mean and standard deviation for normalizing image
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

def runBatchTestFoolX(network, epsilon, csvname):

    #Define variables for results
    foolXApproach_Testing_Results = ""
    Accuracy = 0
    foolXAccuracy = 0
    foolXAvgFk = 0
    foolXAvgDiff = 0
    foolXAvgFroDiff = 0
    fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Perturbed Label', 'Memory Usage', 'Iterations', 'Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']

    counter = 0

    #Create csvwriter for each csv file

    with open(csvname, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    #Open ILSVRC label file
    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')


    for filename in glob.glob('D:/Imagenet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'): #assuming jpg
      print(" \n\n\n**************** FoolX *********************\n" )
      im_orig=Image.open(filename).convert('RGB')
      print(filename)
      if counter == 5000:
          break
      im = transforms.Compose([
          transforms.Scale(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=mean,
                               std=std)])(im_orig)

      def clip_tensor(A, minv, maxv):
          A = torch.max(A, minv * torch.ones(A.shape))
          A = torch.min(A, maxv * torch.ones(A.shape))
          return A

      clip = lambda x: clip_tensor(x, 0, 255)

      imagetransform = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                           transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                           transforms.Lambda(clip)])

      tensortransform = transforms.Compose([transforms.Scale(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                            transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                            transforms.Lambda(clip)])

      start_time = time.time()
      #Input values to hybrid method
      r, loop_i, label_orig, label_pert, pert_image, newf_k = foolx(im, network, epsilon)
      print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
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

      if (int(label_orig) == int(correct)):
          print("Original classification is correct")
          Accuracy = Accuracy + 1

      #If perturbed label matches correct label, add to accuracy count
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
      diff = imagetransform(pert_image.cpu()[0]) - tensortransform(im_orig)
      fro = np.linalg.norm(diff.numpy())
      average = torch.mean(torch.abs(diff))
      foolXAvgFk = foolXAvgFk + newf_k
      foolXAvgDiff = foolXAvgDiff + average
      foolXAvgFroDiff = foolXAvgFroDiff + fro
      rows = []
      rows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert, torch.cuda.memory_stats('cuda:0')['active.all.current'], str(loop_i), str(execution_time), newf_k, average, fro])
      with open(csvname, 'a', newline='') as csvfile:
          csvwriter = csv.writer(csvfile)

          csvwriter.writerows(rows)

      print("#################################### END Hybrid Testing ############################################################\n")
      counter = counter+1

    #Add total values to csv file
    with open(csvname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(epsilon)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(foolXAccuracy/5000)])
        csvwriter.writerows(["Avg F_k: " + str(foolXAvgFk/5000)])
        csvwriter.writerows(["Avg Difference: " + str(foolXAvgDiff / 5000)])
        csvwriter.writerows(["Avg Frobenius of Difference: " + str(foolXAvgFroDiff/5000)])

def runBatchTestDeepfool(network, csvname):
    Accuracy = 0
    DeepfoolAccuracy = 0
    DeepfoolAvgFk = 0
    DeepfoolAvgDiff = 0
    DeepfoolAvgFroDiff = 0
    fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Perturbed Label',
                  'Memory Usage', 'Iterations', 'Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']

    counter = 0

    # Create csvwriter for each csv file

    with open(csvname, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    # Open ILSVRC label file
    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

    for filename in glob.glob(
            'D:/Imagenet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        print(" \n\n\n**************** DeepFool *********************\n")
        im_orig = Image.open(filename).convert('RGB')
        print(filename)
        if counter == 5000:
            break
        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])(im_orig)

        def clip_tensor(A, minv, maxv):
            A = torch.max(A, minv * torch.ones(A.shape))
            A = torch.min(A, maxv * torch.ones(A.shape))
            return A

        clip = lambda x: clip_tensor(x, 0, 255)

        imagetransform = transforms.Compose(
            [transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
             transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
             transforms.Lambda(clip)])

        tensortransform = transforms.Compose([transforms.Scale(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0, 0, 0],
                                                                   std=list(map(lambda x: 1 / x, std))),
                                              transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                              transforms.Lambda(clip)])

        start_time = time.time()
        # Input values to deepfool method
        r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool(im, network)
        print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
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

        if (int(label_orig) == int(correct)):
            print("Original classification is correct")
            Accuracy = Accuracy + 1

        # If perturbed label matches correct label, add to accuracy count
        if (int(label_pert) == int(correct)):
            print("Classifier is correct")
            DeepfoolAccuracy = DeepfoolAccuracy + 1

        def clip_tensor(A, minv, maxv):
            A = torch.max(A, minv * torch.ones(A.shape))
            A = torch.min(A, maxv * torch.ones(A.shape))
            return A

        clip = lambda x: clip_tensor(x, 0, 255)

        print("Iterations: " + str(loop_i))
        diff = imagetransform(pert_image.cpu()[0]) - tensortransform(im_orig)
        fro = np.linalg.norm(diff.numpy())
        average = torch.mean(torch.abs(diff))
        DeepfoolAvgFk = DeepfoolAvgFk + newf_k
        DeepfoolAvgDiff = DeepfoolAvgDiff + average
        DeepfoolAvgFroDiff = DeepfoolAvgFroDiff + fro
        rows = []
        rows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert,
                     torch.cuda.memory_stats('cuda:0')['active.all.current'], str(loop_i), str(execution_time),
                     newf_k, average, fro])
        with open(csvname, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(rows)

        print(
            "#################################### END DeepFool Testing ############################################################\n")
        counter = counter + 1

    # Add total values to csv file
    with open(csvname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(DeepfoolAccuracy / 5000)])
        csvwriter.writerows(["Avg F_k: " + str(DeepfoolAvgFk / 5000)])
        csvwriter.writerows(["Avg Difference: " + str(DeepfoolAvgDiff / 5000)])
        csvwriter.writerows(["Avg Frobenius of Difference: " + str(DeepfoolAvgFroDiff / 5000)])

def runBatchTestFGSM(network, eps, csvname):
    Accuracy = 0
    FGSMAccuracy = 0
    FGSMAvgFk = 0
    FGSMAvgDiff = 0
    FGSMAvgFroDiff = 0
    fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Perturbed Label',
                  'Memory Usage', 'Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']

    counter = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    classifier = art.estimators.classification.PyTorchClassifier(
        model=network,
        input_shape=(3, 224, 224),
        loss=criterion,
        optimizer=optimizer,
        nb_classes=1000
    )
    # Create csvwriter for each csv file

    with open(csvname, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    # Open ILSVRC label file
    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

    for filename in glob.glob(
            'D:/Imagenet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        print(" \n\n\n**************** FGSM *********************\n")
        im_orig = Image.open(filename).convert('RGB')
        print(filename)
        if counter == 5000:
            break
        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])(im_orig)

        def clip_tensor(A, minv, maxv):
            A = torch.max(A, minv * torch.ones(A.shape))
            A = torch.min(A, maxv * torch.ones(A.shape))
            return A

        clip = lambda x: clip_tensor(x, 0, 255)

        imagetransform = transforms.Compose(
            [transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
             transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
             transforms.Lambda(clip)])

        tensortransform = transforms.Compose([transforms.Scale(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0, 0, 0],
                                                                   std=list(map(lambda x: 1 / x, std))),
                                              transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                              transforms.Lambda(clip)])

        start_time = time.time()
        input_batch = im.unsqueeze(0)
        result = classifier.predict(input_batch, 1, False)
        label_orig = np.argmax(result.flatten())
        attack = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=eps, norm=np.inf)
        input_array = input_batch.numpy()
        img_adv = attack.generate(x=input_array)
        print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
        result_adv = classifier.predict(img_adv, 1, False)
        label_pert = np.argmax(result_adv.flatten())
        end_time = time.time()
        execution_time = end_time - start_time
        print("execution time = " + str(execution_time))

        perturbed_img = img_adv.squeeze()
        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]

        str_label_correct = labels[np.int(correct)].split(',')[0]
        str_label_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_pert = labels[np.int(label_pert)].split(',')[0]

        print("Correct label = ", str_label_correct)
        print("Original label = ", str_label_orig)
        print("Perturbed label = ", str_label_pert)
        if (int(label_orig) == int(correct)):
            print("Original classification is correct")

            Accuracy = Accuracy + 1
        pert_image = torch.from_numpy(perturbed_img)

        # If perturbed label matches correct label, add to accuracy count
        if (int(label_pert) == int(correct)):
            print("Classifier is correct")
            FGSMAccuracy = FGSMAccuracy + 1
        diff = imagetransform(pert_image.cpu()) - tensortransform(im_orig)
        fro = np.linalg.norm(diff.numpy())
        average = torch.mean(torch.abs(diff))
        inp = torch.autograd.Variable(torch.from_numpy(input_array[0]).to('cuda:0').float().unsqueeze(0), requires_grad=True)
        fs = network.forward(inp)
        f_k = (fs[0, label_pert] - fs[0, int(correct)]).data.cpu().numpy()
        FGSMAvgFk = FGSMAvgFk + f_k
        FGSMAvgDiff = FGSMAvgDiff + average
        FGSMAvgFroDiff = FGSMAvgFroDiff + fro
        print(FGSMAvgFk)
        rows = []
        rows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert,
                     torch.cuda.memory_stats('cuda:0')['active.all.current'], str(execution_time),
                     f_k, average, fro])
        with open(csvname, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(rows)

        print(
            "#################################### END FGSM Testing ############################################################\n")
        counter = counter + 1

    # Add total values to csv file

    with open(csvname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(FGSMAccuracy / 5000)])
        csvwriter.writerows(["Avg F_k: " + str(FGSMAvgFk / 5000)])
        csvwriter.writerows(["Avg Difference: " + str(FGSMAvgDiff / 5000)])
        csvwriter.writerows(["Avg Frobenius of Difference: " + str(FGSMAvgFroDiff / 5000)])


def runBatchTestFGSM_nonART(network, eps, csvname):
    Accuracy = 0
    FGSMAccuracy = 0
    FGSMAvgFk = 0
    FGSMAvgDiff = 0
    FGSMAvgFroDiff = 0
    fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Perturbed Label',
                  'Memory Usage', 'Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']

    counter = 0

    # Create csvwriter for each csv file

    with open(csvname, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    # Open ILSVRC label file
    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

    for filename in glob.glob(
            'D:/Imagenet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        print(" \n\n\n**************** FGSM *********************\n")
        im_orig = Image.open(filename).convert('RGB')
        print(filename)
        if counter == 5000:
            break
        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])(im_orig)

        def clip_tensor(A, minv, maxv):
            A = torch.max(A, minv * torch.ones(A.shape))
            A = torch.min(A, maxv * torch.ones(A.shape))
            return A

        clip = lambda x: clip_tensor(x, 0, 255)

        imagetransform = transforms.Compose(
            [transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
             transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
             transforms.Lambda(clip)])

        tensortransform = transforms.Compose([transforms.Scale(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0, 0, 0],
                                                                   std=list(map(lambda x: 1 / x, std))),
                                              transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                              transforms.Lambda(clip)])

        start_time = time.time()
        input_batch = im.unsqueeze(0)
        input_array = input_batch.numpy()
        inp = torch.autograd.Variable(torch.from_numpy(input_array[0]).to('cuda:0').float().unsqueeze(0),
                                      requires_grad=True)
        out = network(inp)
        criterion = torch.nn.CrossEntropyLoss()
        pred = np.argmax(out.data.cpu().numpy())
        loss = criterion(out, torch.autograd.Variable(torch.Tensor([float(pred)]).to('cuda:0').long()))

        # compute gradients
        loss.backward()
        grad_orig = inp.grad.data.cpu().numpy().copy()

        # this is it, this is the method
        inp.data = inp.data + (eps * torch.sign(inp.grad.data))
        inp.grad.data.zero_()  # unnecessary

        #print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
        #label_pert = np.argmax(result_adv.flatten())
        end_time = time.time()
        execution_time = end_time - start_time
        print("execution time = " + str(execution_time))

        pred_adv = np.argmax(network(inp).data.cpu().numpy())

        #perturbed_img = img_adv.squeeze()
        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]

        str_label_correct = labels[np.int(correct)].split(',')[0]
        str_label_orig = labels[np.int(pred)].split(',')[0]
        str_label_pert = labels[np.int(pred_adv)].split(',')[0]

        print("Correct label = ", str_label_correct)
        print("Original label = ", str_label_orig)
        print("Perturbed label = ", str_label_pert)
        if (int(pred) == int(correct)):
            print("Original classification is correct")

            Accuracy = Accuracy + 1
        #pert_image = torch.from_numpy(perturbed_img)

        # If perturbed label matches correct label, add to accuracy count
        if (int(pred_adv) == int(correct)):
            print("Classifier is correct")
            FGSMAccuracy = FGSMAccuracy + 1
        adv = inp.data.cpu().numpy()[0]
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        print(tensortransform(im_orig).shape)
        print(adv.shape)
        adv1 = adv.transpose(2,1,0)
        diff = imagetransform(inp.data.cpu()[0]) - tensortransform(im_orig)
        fro = np.linalg.norm(diff.numpy())
        average = torch.mean(torch.abs(diff))
        inp = torch.autograd.Variable(torch.from_numpy(input_array[0]).to('cuda:0').float().unsqueeze(0),
                                      requires_grad=True)
        fs = network.forward(inp)
        f_k = (fs[0, pred_adv] - fs[0, int(correct)]).data.cpu().numpy()
        FGSMAvgFk = FGSMAvgFk + f_k
        FGSMAvgDiff = FGSMAvgDiff + average
        FGSMAvgFroDiff = FGSMAvgFroDiff + fro
        rows = []
        rows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert,
                     torch.cuda.memory_stats('cuda:0')['active.all.current'], str(execution_time),
                     f_k, average, fro])
        with open(csvname, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(rows)

        print(
            "#################################### END FGSM Testing ############################################################\n")
        counter = counter + 1

    # Add total values to csv file

    with open(csvname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(FGSMAccuracy / 5000)])
        csvwriter.writerows(["Avg F_k: " + str(FGSMAvgFk / 5000)])
        csvwriter.writerows(["Avg Difference: " + str(FGSMAvgDiff / 5000)])
        csvwriter.writerows(["Avg Frobenius of Difference: " + str(FGSMAvgFroDiff / 5000)])


#runBatchTestFoolX(netr101, 0.0005, 'resnet1010005.csv')
#runBatchTestFoolX(netr101, 0.005, 'resnet101005linf.csv')
#runBatchTestFoolX(netr101, 0.05, 'resnet10105.csv')
#runBatchTestFoolX(netr101, 0.001, 'resnet101001.csv')
#runBatchTestFoolX(netr101, 0.01, 'resnet10101.csv')
#runBatchTestFoolX(netr34, 0.0005, 'resnet340005.csv')
#runBatchTestFoolX(netr34, 0.005, 'resnet34005linf.csv')
#runBatchTestFoolX(netr34, 0.05, 'resnet3405.csv')
#runBatchTestFoolX(netr34, 0.001, 'resnet34001.csv')
#runBatchTestFoolX(netr34, 0.01, 'resnet3401.csv')
#runBatchTestFoolX(neta, 0.0005, 'alexnet0005.csv')
#runBatchTestFoolX(neta, 0.005, 'alexnet005linf.csv')
#runBatchTestFoolX(neta, 0.05, 'alexnet05.csv')
#runBatchTestFoolX(neta, 0.001, 'alexnet001.csv')
#runBatchTestFoolX(neta, 0.01, 'alexnet01.csv')
#runBatchTestFoolX(netg, 0.0005, 'googlenet0005.csv')
#runBatchTestFoolX(netg, 0.005, 'googlenet005linf.csv')
#runBatchTestFoolX(netg, 0.05, 'googlenet05.csv')
#runBatchTestFoolX(netg, 0.001, 'googlenet001.csv')
#runBatchTestFoolX(netg, 0.01, 'googlenet01.csv')
#runBatchTestFoolX(netr101, 0.1, 'resnet1011.csv')
#runBatchTestFoolX(netr34, 0.1, 'resnet341.csv')
#runBatchTestFoolX(neta, 0.1, 'alexnet1.csv')
#runBatchTestFoolX(netg, 0.1, 'googlenet1.csv')

runBatchTestFGSM_nonART(netr34, 0.0005, 'resnet340005.csv')
runBatchTestFGSM_nonART(netr34, 0.005, 'resnet34005.csv')
runBatchTestFGSM_nonART(netr34, 0.05, 'resnet3405.csv')
runBatchTestFGSM_nonART(netr34, 0.001, 'resnet34001.csv')
runBatchTestFGSM_nonART(netr34, 0.01, 'resnet3401.csv')
runBatchTestFGSM_nonART(netr34, 0.1, 'resnet341.csv')

runBatchTestFGSM_nonART(netr101, 0.0005, 'resnet1010005.csv')
runBatchTestFGSM_nonART(netr101, 0.005, 'resnet101005.csv')
runBatchTestFGSM_nonART(netr101, 0.05, 'resnet10105.csv')
runBatchTestFGSM_nonART(netr101, 0.001, 'resnet101001.csv')
runBatchTestFGSM_nonART(netr101, 0.01, 'resnet10101.csv')
runBatchTestFGSM_nonART(netr101, 0.1, 'resnet1011.csv')

runBatchTestFGSM_nonART(neta, 0.0005, 'alexnet0005.csv')
runBatchTestFGSM_nonART(neta, 0.005, 'alexnet005.csv')
runBatchTestFGSM_nonART(neta, 0.05, 'alexnet05.csv')
runBatchTestFGSM_nonART(neta, 0.001, 'alexnet001.csv')
runBatchTestFGSM_nonART(neta, 0.01, 'alexnet01.csv')
runBatchTestFGSM_nonART(neta, 0.1, 'alexnet1.csv')

runBatchTestFGSM_nonART(netg, 0.0005, 'googlenet0005.csv')
runBatchTestFGSM_nonART(netg, 0.005, 'googlenet005.csv')
runBatchTestFGSM_nonART(netg, 0.05, 'googlenet05.csv')
runBatchTestFGSM_nonART(netg, 0.001, 'googlenet001.csv')
runBatchTestFGSM_nonART(netg, 0.01, 'googlenet01.csv')
runBatchTestFGSM_nonART(netg, 0.1, 'googlenet1.csv')



