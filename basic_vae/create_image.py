import torch
import argparse
from torchvision.utils import save_image
from models import *
from datetime import datetime

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
model = VAE().to(device)

PATH = "/Users/mohsenk/Downloads/vae_models/model.pth"
model.load_state_dict(torch.load(PATH))

if __name__ == "__main__":
    current_time_str = "{}{}{}{}{}".format(datetime.now().month,
                                         datetime.now().day,
                                         datetime.now().hour,
                                         datetime.now().minute,
                                         datetime.now().second)
   
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        one_sample = sample[0,:].view(1, 1, 28, 28)
        print(sample.size())
        print(one_sample.size())
        save_image(sample.view(64, 1, 28, 28),
                   '/Users/mohsenk/Downloads/vae_mohsen_samples/batch_sample_' + current_time_str + '.png')
        save_image(one_sample,
                   '/Users/mohsenk/Downloads/vae_mohsen_samples/one_sample_' + current_time_str + '.png')
