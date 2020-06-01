from argparse import ArgumentParser
from PIL import Image, ImageFilter
from unet import *
import torchvision.transforms as tvF
from check import add_text

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of TextRemoval')

    # Data parameters
    parser.add_argument('-a', '--add-text', help='add text on the image', action='store_true')
    parser.add_argument('-d', '--test-dir', help='test set path', default='./data/test')
    parser.add_argument('-n', '--imge-name', help='test image saved in test-dir')
    parser.add_argument('-p', '--noise-param', help='noise parameter (largest coverage for text)', default=0.5, type=float)
    parser.add_argument('-m', '--model', help='model chosen to test', default='./ckpts/best.pt')
    #parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--pre-set', help='pre-processing the image by shrink and sharpen the image', default=1, type=int)
    parser.add_argument('-s', '--save-path', help='path to save the image', default='./data/test_save')
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    source = params.test_dir
    if source[-1] != '/':
        source += '/'
    save_path = params.save_path
    if save_path[-1] != '/':
        save_path += '/'
    source += params.imge_name
    print(source)
    model_path = params.model
    unet = UNet()
    unet.load_state_dict(torch.load('ckpts/best.pt', map_location='cpu'))
    
    if params.add_text:
        imge = add_text(source, params.noise_param)
    else:
        imge = Image.open(source).convert('RGB')
    if params.pre_set != 1:
        w, h = imge.size
        imge = imge.resize((int(w/params.pre_set), int(h/params.pre_set)), Image.ANTIALIAS)
        imge = imge.filter(ImageFilter.SHARPEN)
    imge = tvF.ToTensor()(imge)
    imge = tvF.Normalize(mean = (0.5,0.5,0.5), std = (1,1,1))(imge)
    imge = imge.unsqueeze(0)
    denoise = unet(imge).detach()
    denoise = denoise.squeeze(0)
    denoise = tvF.Normalize(mean = (-0.5,-0.5,-0.5), std = (1,1,1))(denoise)
    denoise = tvF.ToPILImage()(denoise)
    denoise.show()
    denoise.save(save_path+params.imge_name)
    