import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2 as cv
from tensorboardX import SummaryWriter
import datetime

def tensor2np(tensor, resize_to=None):

    out_array = tensor.detach().cpu().numpy()
    out_array = np.moveaxis(out_array, 0, -1)    # (CHW) -> (HWC)

    if resize_to is not None:
        out_array = cv.resize(out_array, dsize=resize_to, interpolation=cv.INTER_CUBIC)
        out_array = np.expand_dims(out_array, axis=0)

    return out_array

class DeepFeatures(torch.nn.Module):

    def __init__(self,  model,
                        tensorboard_folder='./Tensorboard',
                        experiment_name='mnist_embeds_net3'):

        super(DeepFeatures, self).__init__()

        self.model = model
        self.model.eval()

        self.tensorboard_folder = tensorboard_folder

        self.name = experiment_name

        self.writer = None

    def generate_embeddings(self, x, normalize=False):

        embeds = self.model.embedding(x)

        if normalize:
            embeds = F.normalize(embeds)

        return embeds

    def write_embeddings(self, x, outsize=(28, 28)):
        embeds = self.generate_embeddings(x)

        embeds = embeds.detach().cpu().numpy()
        # embeds = embeds/np.linalg.norm(embeds)  # normalize to visualize embeds around unit sphere

        for i in range(len(embeds)):
            key = str(np.random.random())[-7:]
            np.save(self.images_folder + r"/" + key + '.npy', tensor2np(x[i], outsize))
            np.save(self.embeds_folder + r"/" + key + '.npy', embeds[i]/np.linalg.norm(embeds[i]))

        return True

    def create_tensorboard_dirs(self, model_type='modified'):
        dt = datetime.datetime.now()
        dt_suffix = '_' + dt.strftime("%x").replace('/', '_') + '_' +  dt.strftime("%X").replace(':', '_')

        if self.name is None:
            name = model_type + '_' + 'Experiment_' + dt_suffix
        else:
            name = model_type + '_' + self.name + dt_suffix

        self.dir_name = os.path.join(self.tensorboard_folder, name)
        self.images_folder = os.path.join(self.dir_name, 'images')
        self.embeds_folder = os.path.join(self.dir_name, 'embeds')

        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
            os.mkdir(self.images_folder)
            os.mkdir(self.embeds_folder)
        else:
            print('Warning: logfile already exists')
            print('logging directory: ' + str(self.dir_name))

    def _create_writer(self):
        self.writer = SummaryWriter(logdir=self.dir_name)
        return True

    def create_tensorboard_log(self):
        if self.writer is None:
            self._create_writer()

        all_embeds = [np.load(os.path.join(self.embeds_folder, path)) for path in os.listdir(self.embeds_folder) if path.endswith('.npy')]
        all_images = [np.load(os.path.join(self.images_folder, path)) for path in os.listdir(self.images_folder) if path.endswith('.npy')]
        # all_images = [np.moveaxis(image, 0, -1) for image in all_images]
        all_embeds = torch.Tensor(all_embeds)
        all_images = torch.Tensor(all_images)

        print(all_embeds.shape)
        print(all_images.shape)

        self.writer.add_embedding(all_embeds, label_img = all_images)
