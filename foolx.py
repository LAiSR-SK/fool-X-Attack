import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients

#Fool-X algorithm to calculate minimum perturbation from maximum hyperplanes.

def foolx(image, net, eps=0.05, num_classes=10, overshoot=0.02, max_iter=50):
    #image - the image to be input into the algorithm, net - the network for perturbations to be generated against,
    #eps - epsilon value, controls the size of the perturbation,
    #num_classes - number of nearby classes that will be used to calculate the perturbation,
    #overshoot - termination value to prevent vanishing updates,
    #max-iter - maximum iterations, if no perturbation is found, terminate after this number of iterations

    #Check if cuda is available.
    is_cuda = torch.cuda.is_available()

    #If cuda is available use GPU for faster processing, if not, use CPU.
    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    #Convert image into tensor readable by PyTorch, flatten image.
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    #Create array of labels.
    label_array = (np.array(f_image)).flatten().argsort()[::-1]

    #Define array as size of specified number of classes, set first class to the original label.
    label_array = label_array[0:num_classes]
    label = label_array[0]

    #Copy the image, create variable for perturbed image, as well as w and r_tot, using the shape of the image
    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    #initialize loop variable to 0
    loop_i = 0

    #Set x to the original image, forward propagate it through the network, get list of classes
    x = Variable(pert_image[None, :], requires_grad=True)
    forward_prop = net.forward(x)
    fp_list = [forward_prop[0, label_array[k]] for k in range(num_classes)]
    current_label = label

    #While label equals original label and max iterations not reached:
    while current_label == label and loop_i < max_iter:

        #Backwards propagate label through graph, get resulting gradient and gradient sign.
        pert = 0  # np.inf we change the to be zero instead of infinty
        forward_prop[0, label_array[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        grad_orig_sign = x.grad.sign().cpu().numpy().copy()  # added for fgsm

        for k in range(1, num_classes):
            zero_gradients(x)

            #Backwards propagate current label through graph, get resulting gradient and gradient sign.
            forward_prop[0, label_array[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()
            cur_sign_grad = x.grad.sign().cpu().numpy().copy()  # added for fgsm

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (forward_prop[0, label_array[k]] - forward_prop[0, label_array[0]]).data.cpu().numpy()

            #Calculate perturbation using deepfool formula
            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten()) #ord=np.inf)

            # determine which w_k to use
            if pert_k > pert:  # we change here the "<" to be ">" to get the max hyperplanes
                pert = pert_k
                w = w_k
            point = k
        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        #combined calculated perturbation with FGSM, multiply the gradient by the epsilon value
        r_tot = np.float32(r_tot + r_i) + eps * cur_sign_grad #w_k_sign

        #If cuda is being used, store perturbed image tensor in GPU, if not, store it in CPU
        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
            pert = ((1 + overshoot) * torch.from_numpy(r_tot)) * 100
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)
            pert = ((1 + overshoot) * torch.from_numpy(r_tot))

        #Add new perturbation to x
        x = Variable(pert_image, requires_grad=True)

        #Propagate x through network, get new label, get new f_k distance.
        forward_prop = net.forward(x)
        current_label = np.argmax(forward_prop.data.cpu().numpy().flatten())
        newf_k = (forward_prop[0, current_label] - forward_prop[0, label_array[0]]).data.cpu().numpy()
        loop_i += 1


    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, current_label, pert_image, pert, newf_k
