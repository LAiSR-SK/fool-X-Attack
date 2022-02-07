import art
from torchvision import models
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

im_orig = Image.open('pictures/test_im2.jpg')
net = models.resnet34(pretrained=True)
net.eval()

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

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

tf_mod = transforms.Compose([transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

classifier = art.estimators.classification.PyTorchClassifier(
    model=net,
    input_shape=(3, 224, 224),
    loss = criterion,
    optimizer=optimizer,
    nb_classes=1000
)

input_tensor = preprocess(im_orig)
input_batch = input_tensor.unsqueeze(0)
print(input_batch.shape)

a = classifier.predict(input_batch, 1, False)
#print(a)

label_orig = np.argmax(a.flatten())
print(label_orig)

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]

print("Original label = ", str_label_orig)

attack = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.2, norm=np.inf)

input_array = input_batch.numpy()
img_adv = attack.generate(x=input_array)

b = classifier.predict(img_adv, 1, False)
label_pert = np.argmax(b.flatten())
str_label_pert = labels[np.int(label_pert)].split(',')[0]
print("Perturbed label = ", str_label_pert)

original_img = input_array.squeeze()
perturbed_img = img_adv.squeeze()
original_img = original_img.swapaxes(0,1) #(3,224,224) -> (224,3,224)
original_img = original_img.swapaxes(1,2) #(224,3,224) -> (224,224,3)
#perturbed_img = perturbed_img.swapaxes(0,1)
#perturbed_img = perturbed_img.swapaxes(1,2)

print(original_img.shape)
print(input_tensor.shape)
print(original_img)
print(input_tensor)
print(label_pert)

inp = torch.autograd.Variable(torch.from_numpy(input_array[0]).to('cpu').float().unsqueeze(0), requires_grad=True)
out = net(inp)
criterion = torch.nn.CrossEntropyLoss()
pred = np.argmax(out.data.cpu().numpy())
loss = criterion(out, torch.autograd.Variable(torch.Tensor([float(pred)]).long()))

# compute gradients
loss.backward()
grad_orig = inp.grad.data.cpu().numpy().copy()

# this is it, this is the method
inp.data = inp.data + (0.2 * torch.sign(inp.grad.data))
inp.grad.data.zero_()  # unnecessary
fs = net.forward(inp)

correct = '150'
pred_adv = np.argmax(net(inp).data.cpu().numpy())
print(pred)
print(pred_adv)
f_k = (fs[0, pred_adv] - fs[0, int(correct)]).data.cpu().numpy()
print(f_k)
adv = inp.data.cpu().numpy()[0]
adv = adv.transpose(1, 2, 0)
adv = (adv * std) + mean
adv = adv * 255.0
adv = np.clip(adv, 0, 255).astype(np.uint8)
#adv = adv.transpose(1, 0, 2) #3,224,224
print(adv.shape)
print(input_tensor.shape)

plt.figure()
plt.imshow(tf(torch.from_numpy(adv)))
plt.title(str_label_pert)
plt.show()


plt.figure()
plt.imshow(tf(input_tensor))
plt.title(str_label_orig)
plt.show()
plt.imshow(tf(torch.from_numpy(perturbed_img)))
plt.title(str_label_pert)
plt.show()

fgsm_p = torch.from_numpy(perturbed_img) - input_tensor
plt.imshow(tf_mod(fgsm_p))
plt.title(str_label_pert)
plt.show()
