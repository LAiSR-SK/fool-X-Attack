
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
from foolx import foolx
import os
import time

#Check if cuda is available.
is_cuda = torch.cuda.is_available()
torch.device('cpu')

#If cuda is available use GPU for faster processing, if not, use CPU.
if is_cuda:
    print("Using GPU")
    torch.device('cuda')
else:
    print("Using CPU")


net = models.resnet34(pretrained=True)
#net = models.alexnet(pretrained=True)
# Switch to evaluation mode
net.eval()
if is_cuda:
    net.cuda()

im_orig = Image.open('new/ILSVRC2017_test_00004324.JPEG').convert('RGB')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

# Remove the mean
im = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)

start_time = time.time()
r, loop_i, label_orig, label_pert, pert_image, pert, newf_k = foolx(im, net, 0.005)
end_time = time.time()
execution_time = end_time - start_time
print("execution time = " + str(execution_time))

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)


tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                        transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

plt.figure()
plt.imshow(tf(im.cpu()))
plt.title(str_label_orig)
tf(im.cpu()).save('orig.png')
plt.show()
plt.figure()
plt.imshow(tf(pert_image.cpu()[0]))
img = tf(pert_image.cpu()[0])
img.save('image.png')
plt.title(str_label_pert)
plt.show()
print(loop_i)
plt.figure()
fc = 1000000
plt.imshow(tf(pert.cpu()[0]*fc))
tf(pert.cpu()[0]*fc).save('perturbation.png')
plt.title(str_label_pert)
plt.show()

#10000000(0) - 0.0005
#1000000(0) - 0.005
#10000 - 0.05
#100 - 0.1
#100000 - 0.01
#10000000 - 0.001
#10000000000 - 0.0001
