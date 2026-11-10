import ast
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models_module
from torch import __version__

# obtain ImageNet labels
classes_file = 'imagenet1000_clsid_to_human.txt'
if not os.path.exists(classes_file):
    classes_file = os.path.join(os.path.dirname(__file__), 'imagenet1000_clsid_to_human.txt')

with open(classes_file) as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

class _LazyModelDict(dict):
    """
    Lazy-loads pre-trained torchvision CNN models on demand.
    Prevents eager download of all model architectures at module import time.
    """
    def __getitem__(self, key):
        if not dict.__contains__(self, key):
            if key == 'resnet':
                dict.__setitem__(self, 'resnet', models_module.resnet18(pretrained=True))
            elif key == 'alexnet':
                dict.__setitem__(self, 'alexnet', models_module.alexnet(pretrained=True))
            elif key == 'vgg':
                dict.__setitem__(self, 'vgg', models_module.vgg16(pretrained=True))
            else:
                raise KeyError(f"Unsupported model architecture: {key}")
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        return key in ('resnet', 'alexnet', 'vgg')

models = _LazyModelDict()

def classifier(img_path, model_name):
    # load the image
    img_pil = Image.open(img_path)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    
    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    
    # wrap input in variable, wrap input in variable - no longer needed for
    # v 0.4 & higher code changed 04/26/2018 by Jennifer S. to handle PyTorch upgrade
    pytorch_ver = __version__.split('.')
    
    # pytorch versions 0.4 & hihger - Variable depreciated so that it returns
    # a tensor. So to address tensor as output (not wrapper) and to mimic the 
    # affect of setting volatile = True (because we are using pretrained models
    # for inference) we can set requires_gradient to False. Here we just set 
    # requires_grad_ to False on our tensor 
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)
    
    # pytorch versions less than 0.4 - uses Variable because not-depreciated
    else:
        # apply model to input
        # wrap input in variable
        data = Variable(img_tensor, volatile = True) 

    # apply model to input
    model = models[model_name]

    # puts model in evaluation mode
    # instead of (default)training mode
    model = model.eval()
    
    # apply data to model - adjusted based upon version to account for 
    # operating on a Tensor for version 0.4 & higher.
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        output = model(img_tensor)

    # pytorch versions less than 0.4
    else:
        # apply data to model
        output = model(data)

    # return index corresponding to predicted class
    pred_idx = output.data.numpy().argmax()

    return imagenet_classes_dict[pred_idx]
