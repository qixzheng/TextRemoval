from argparse import ArgumentParser
from PIL import Image, ImageFilter
from unet import *
import torchvision.transforms as tvF

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of TextRemoval')

    # Data parameters
    parser.add_argument('-a', '--add-text', help='add text on the image', action='stroe_true')
    parser.add_argument('-d', '--test-dir', help='test set path', default='./data/test')
    parser.add_argument('-n', '--imge-name', help='test image saved in test-dir')
    parser.add_argument('-m', '--model', help='model chosen to test')
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('-s', '--save-path', help='path to save the image', default='./data/test_save')
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    source = params.test_dir
    if source[-1] != '/':
        source += '/'
    source += params.imge_name
    model_path = params.model
    unet = UNet()
    unet.load_state_dict(torch.load('ckpts/best.pt', map_location='cpu'))
    imge = Image.open(source).convert('RGB')
    #imge = imge.resize((683, 384),Image.ANTIALIAS)
    #imge = imge.filter(ImageFilter.SHARPEN)
    imge = tvF.ToTensor()(imge)
    imge = tvF.Normalize(mean = (0.5,0.5,0.5), std = (1,1,1))(imge)
    imge = imge.unsqueeze(0)
    denoise = unet(imge).detach()
    denoise = denoise.squeeze(0)
    denoise = tvF.Normalize(mean = (-0.5,-0.5,-0.5), std = (1,1,1))(denoise)
    denoise = tvF.ToPILImage()(denoise)
    denoise.show()
    denoise.save('true.bmp')
    