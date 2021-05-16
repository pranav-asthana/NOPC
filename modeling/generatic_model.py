import torch
from .UNet import UNet
from .PCPR import PCPRender
from .PCPR import PCPRParameters
from .PointGenerator import PCGenerator




class Generatic_Model(torch.nn.Module):


    def __init__(self, tar_width, tar_height, feature_dim, dataset=None, use_rgb = False ):
        super(Generatic_Model, self).__init__()
        self.dataset = dataset
        # self.pcpr_parameters = PCPRParameters(vertex_list, feature_dim) # We don't want to use this
        ## Add our PCGenerator network
        self.pc_generator = PCGenerator(N=1024, k=feature_dim)
        print(self.pc_generator)
        self.render = PCPRender(feature_dim,tar_width,tar_height, dataset = dataset)

        input_channels = 0
        self.use_rgb = use_rgb
        if use_rgb:
            input_channels = 3


    def forward(self, images, K, T,
           near_far_max_splatting_size, inds=None):


        # num_points = num_points.int()


        # print(point_indexes) # Not sure what this is
        # print(in_points.shape) # We will not use this, to be ignored
        # print(K.shape)
        # print(T.shape)
        # print(near_far_max_splatting_size) # Need to figure out what this is
        # print(num_points) # Again not sure. why are there 3? This is important as we need to set it manually for our model
        # print(rgbs.shape) # RGB values for each point in in_points. Need to be ignored
        '''
        [0, 0, 0]
        torch.Size([969885, 3])
        torch.Size([3, 3, 3])
        torch.Size([3, 4, 4])
        tensor([[ 6., 16.,  1.],
                [ 6., 16.,  1.],
                [ 6., 16.,  1.]])
        tensor([323295, 323295, 323295], dtype=torch.int32)
        torch.Size([969885, 0])
        '''



        # _,default_features,_ = self.pcpr_parameters(point_indexes)

        ## We want the features to come from out E-D network, not here. The E-D needs to be added in this file
        # p_parameters,default_features,_ = self.pcpr_parameters(point_indexes)
        

        # Not sure what default_features does
        # TODO: Need to get input images here to be fed into network, not points
        # print(in_points.shape, p_parameters.shape)
        print(images[0].shape)
        in_points, p_parameters, default_features = self.pc_generator((images[0],))

        # print(in_points.shape, p_parameters.shape)
        
        batch_size = K.size(0)
        dim_features = default_features.size(0)

        num_points = torch.tensor([4096]*batch_size)

        m_point_features = []
        beg = 0

        for i in range(batch_size):
            
            m_point_features.append(p_parameters)

        point_features = torch.cat(m_point_features, dim = 1).requires_grad_()
        

        res,depth,features,dir_in_world, rgb = self.render(point_features, default_features,
                             in_points,
                             K, T,
                             near_far_max_splatting_size, num_points, inds)

        
        return res,depth, features, dir_in_world, rgb, point_features

        
