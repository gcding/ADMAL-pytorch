import torch
import torchvision.transforms as standard_transforms
from torch.autograd import Variable

from networks.Counter import Counter

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from nni.compression.pytorch.utils.counter import count_flops_params

import getopt
import sys
import math

arguments_strModel = "ADML"
arguments_strMode = 'DME'
arguments_strModelStateDict = './weights/adml_small_vehicle.pth'
arguments_strImg = './image/P2068.png'
arguments_strOut = './out/1_out.png'

arguments_intDevice = 0

mean_std =  ([0.36475515365600586, 0.36875754594802856, 0.34205102920532227], [0.2001768797636032, 0.19185248017311096, 0.1892034411430359])

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

for strOption, strArgument in \
getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--device' and strArgument != '': arguments_intDevice = int(strArgument)  # device number
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # model type
    if strOption == '--mode' and strArgument != '': arguments_strMode = strArgument  # mode type
    if strOption == '--model_state' and strArgument != '': arguments_strModelStateDict = strArgument  # path to the model state
    if strOption == '--img_path' and strArgument != '': arguments_strImg = strArgument  # path to the image
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored

torch.cuda.set_device(arguments_intDevice)

#  If you want to test the average inference speed, please rewrite it yourself to read the images of the whole dataset.
def evaluate(img_path, save_path):
    if arguments_strModel == "ADML":
        net = Counter(model_name = arguments_strModel, mode = arguments_strMode)
        net.load_state_dict(torch.load(arguments_strModelStateDict, map_location=lambda storage, loc: storage), strict=False)
        net.cuda()
        net.eval()
        # Flops  is calculated with an image size of 224*224.
        # img_test = torch.randn(1, 3, 224, 224)
        # img_test_1 = torch.randn(1, 1, 224, 224)
        # flops, params, results = count_flops_params(net, (img_test, img_test_1))
        # print(params / (1000*1000), flops / (1000*1000*1000))
    elif arguments_strModel == "ASPDNet":
        net = Counter(model_name = arguments_strModel, mode = arguments_strMode)
        net.load_state_dict(torch.load(arguments_strModelStateDict, map_location=lambda storage, loc: storage), strict=False)
        net.cuda()
        net.eval()
    else:
        raise ValueError('Network cannot be recognized. Please define your own Network here.')
    
    img = Image.open(img_path)

    if img.mode == "L":
        img = img.convert('RGB')

    img = img_transform(img)

    img = img.view(1, img.size(0), img.size(1), img.size(2))
    w = math.ceil(img.shape[2] / 16) * 16
    h = math.ceil(img.shape[3] / 16) * 16
    data_list = torch.FloatTensor(1,3,int(w),int(h)).fill_(0)
    data_list[:,:,0:img.shape[2],0:img.shape[3]] = img
    img = data_list

    with torch.no_grad():
        
        img = Variable(img).cuda()
        torch.cuda.synchronize() 
        start_time = time.time()
        pred_map = net.test_forward(img, None)
        torch.cuda.synchronize()
        end_time = time.time()
        diff_time = start_time - end_time

    pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
    pred = np.sum(pred_map)/100.0
    pred_map = pred_map/np.max(pred_map+1e-20)

    print("count result is {}".format(pred))

    den_frame = plt.gca()
    plt.imshow(pred_map, 'jet')
    plt.colorbar()
    den_frame.axes.get_yaxis().set_visible(False)
    den_frame.axes.get_xaxis().set_visible(False)
    den_frame.spines['top'].set_visible(False) 
    den_frame.spines['bottom'].set_visible(False) 
    den_frame.spines['left'].set_visible(False) 
    den_frame.spines['right'].set_visible(False) 
    plt.savefig(save_path, bbox_inches='tight',pad_inches=0,dpi=150)
    plt.close()

    print("save pred density map in {} success".format(arguments_strOut))

    print("end")


if __name__ == '__main__':
    evaluate(arguments_strImg, arguments_strOut)
