import numpy as np
import torch
import torch.nn as nn

class PCGenerator(nn.Module):
    def __init__(self, images = 1, N=1024, k=1024):
        '''
        Number of input images
        N = # points to generate
        k = # features attached to each point
        '''
        super().__init__()
        self.k = k
        self.default_features = torch.nn.Parameter(torch.randn(k, 1))

        self.f_ext = [nn.Conv2d(3, 64, 3, padding=3//2), nn.ReLU()]
        self.f_ext += [nn.Conv2d(64, 64, 3, padding=3//2), nn.ReLU()]
        self.f_ext += [nn.MaxPool2d(2)]
        self.f_ext += [nn.Dropout(0.25)]

        self.f_ext += [nn.Conv2d(64, 128, 3, padding=3//2), nn.ReLU()]
        self.f_ext += [nn.Conv2d(128, 128, 3, padding=3//2), nn.ReLU()]
        self.f_ext += [nn.MaxPool2d(2)]
        self.f_ext += [nn.Dropout(0.25)]

        self.f_ext += [nn.Conv2d(128, 256, 3, padding=3//2), nn.ReLU()]
        self.f_ext += [nn.Conv2d(256, 256, 3, padding=3//2), nn.ReLU()]
        self.f_ext += [nn.MaxPool2d(4)]
        self.f_ext += [nn.Dropout(0.25)]

        ########

        # TODO: That number below depends on the input image dimension. Need to change. This is only for 200*200 images
        self.fc_layers = [nn.Linear(36864*images, k*4), nn.ReLU()]
        self.fc_layers += [nn.Linear(k*4, k*4), nn.ReLU()]

        ########

        self.decoder = [nn.ConvTranspose2d(k*4, 3 + k*2, 3), nn.ReLU()] # The 3 is for xyz position of point and k is for features
        self.decoder += [nn.ConvTranspose2d(3 + k*2, 3 + k*2, 3), nn.ReLU()] 
        self.decoder += [nn.Upsample(scale_factor=2)]

        self.decoder += [nn.ConvTranspose2d(3 + k*2, 3 + k, 3), nn.ReLU()]
        self.decoder += [nn.ConvTranspose2d(3 + k, 3 + k, 3), nn.ReLU()]
        self.decoder += [nn.Upsample(scale_factor=2)]

        self.decoder += [nn.ConvTranspose2d(3 + k, 3 + k, 3), nn.ReLU()]
        self.decoder += [nn.ConvTranspose2d(3 + k, 3 + k, 3), nn.ReLU()]
        self.decoder += [nn.Upsample(scale_factor=2)]

     
        self.f_ext = nn.ModuleList(self.f_ext)
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.decoder = nn.ModuleList(self.decoder)

    def extract_features(self, x):
      f = x
      for i in range(len(self.f_ext)):
        f = self.f_ext[i](f)
      return f

    def forward(self, x):
      # x is of shape (batch_size, channels=3, width, height)
      feature_vectors = [self.extract_features(xi).flatten() for xi in x]
      feature_vectors = torch.cat(feature_vectors)
      
      # TODO: concat feature vectors with camera params into variable z
      # Do we need to or not? Try without first
      z = feature_vectors
      print(z.shape)

      for i in range(len(self.fc_layers)):
        z = self.fc_layers[i](z)

      # z is the latent code which is fed into the generator
      p = z.reshape(1, self.k*4, 1, 1)
      for i in range(len(self.decoder)):
        p = self.decoder[i](p)
      # p is the final pointset
      p = p.reshape(3 + self.k, -1).T
      positions = p[:, :3].contiguous()
      features = p[:, 3:].T.contiguous()
      return positions, features, self.default_features


# Testing
# model = PCGenerator().float().to(device)
# print(model)
# im1 = np.random.rand(1, 3, 200, 200)
# im1 = torch.from_numpy(im1).float().to(device)


# print(model((im1,)).shape)
# # del model
# # torch.cuda.empty_cache()