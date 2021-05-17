# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import torch

from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage

from apex import amp



def create_supervised_trainer(model, optimizer, loss_fn, use_cuda=True, swriter = None):

    if use_cuda:
        model.cuda()

    global iters
    iters = 0


    def res_hook2(grad):
        global iters
        swriter.add_scalars('gradient2', {'res_max':torch.max(grad).item(),
                        'res_min':torch.min(grad).item()}, iters)

    def res_hook(grad):
        global iters
        swriter.add_scalars('gradient', {'res_max':torch.max(grad).item(),
                        'res_min':torch.min(grad).item()}, iters)
    
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()

        # print(len(batch))
        

        # in_points = batch[1].cuda()
        # K = batch[2].cuda() # intrinsic
        # T = batch[3].cuda() # extrinsic
        # near_far_max_splatting_size = batch[5]
        # num_points = batch[4]
        # point_indexes = batch[0]
        # target = batch[7].cuda()
        # inds = batch[6].cuda() # Blacklist indexes, can be ignored
        # rgbs = batch[8].cuda()

        target = batch[0][0].cuda()
        K = batch[1][0].cuda()
        T = batch[2][0].cuda()
        near_far_max_splatting_size = batch[3].cuda()
        label = batch[4][0]
        inds = None

        near_far_max_splatting_size = near_far_max_splatting_size.repeat(K.shape[0], 1)

        # print('DEBUG: Target shape: {}'.format(target.shape))
        # print('DEBUG: K shape: {}'.format(K.shape))
        # print('DEBUG: T shape: {}'.format(T.shape))
        # print('DEBUG: nearfar_max_splat_size: {} | shape: {}'.format(near_far_max_splatting_size, near_far_max_splatting_size.shape))
        # print('DEBUG: label: ', label)

        res,depth,features,dir_in_world,rgb,m_point_features = model(target[:1, :3, :, :], K, T,
                            near_far_max_splatting_size, inds)

        # print('DEBUG: res shape: ', res.shape)
        # print('DEBUG: depth shape: ', depth.shape)
        # print('DEBUG: features shape: ', features.shape)
        # print('DEBUG: m_point_features shape: ', m_point_features.shape)

        # TODO: Change how losses are being calculated and handled
        '''
        1. We feed 1 image into PCGenerator to get a PC
        2. We feed that same camera pose to get the rendered image and calculate loss (loss1)
        3. We feed additional 4 camera poses to get rendered images and caluclate those losses (loss2 to loss5)
        4. We use each individual loss(loss1 to loss5) to update the Unet portion of the pipeline
        5. We use the weighted sum(for now equal weight) of losses to update the PCGenerator portion of pipeline
        6. Repeat
        '''

        res.register_hook(res_hook2)
        m_point_features.register_hook(res_hook)

        # depth_mask = depth[:,0:1,:,:].detach().clone()
        # depth_mask = depth_mask + target[:,3:4,:,:]
        # depth_mask[depth_mask>0] = 1
        # depth_mask = torch.clamp(depth_mask,0,1.0)


        # cur_mask = (target[:,4:5,:,:]*depth_mask).repeat(1,3,1,1)


        # target[:,0:3,:,:] = target[:,0:3,:,:]*cur_mask
        # res[:,0:3,:,:] = res[:,0:3,:,:]*cur_mask



        # vis_rgb = res[0,0:3,:,:].detach().clone()
        # vis_rgb[cur_mask[0]<0.1] += 0.5


        # render_mask = res[0][3:4,:,:].detach().clone()
        # render_mask[target[0,4:5,:,:]<0.1] += 0.5

        # render_mask = torch.clamp(render_mask,0,1.0)


        # res[:,3,:,:] = res[:,3,:,:] * target[:,4,:,:]

        ## New loss calculations
        # Skip all this stuff above since we are not doing any transformations to data currently
        # print('DEBUG: result shape: {}'.format(res.shape))
        # print('DEBUG: target shape: {}'.format(target.shape))

        loss1, loss2, loss3 = loss_fn(res, target[:,0:4,:,:]) # Loss over all 5 images


        l = loss1 + loss2 # Reconstruction + perceptual

        l.backward()
        # with amp.scale_loss(l, optimizer) as scaled_loss:
        #     scaled_loss.backward()


        optimizer.step()

        iters = engine.state.iteration

        swriter.add_scalar('Loss/train_ploss',loss1.item(), iters)
        swriter.add_scalar('Loss/train_mseloss',loss2.item(), iters)
        swriter.add_scalar('Loss/train_gradloss',loss3.item(), iters)
        swriter.add_scalar('Loss/train_totalloss',l.item(), iters)

        #swriter.add_scalars('gradient', {'p-features_max':torch.max(m_point_features.grad).item(),
        #                'p-features_min':torch.min(m_point_features.grad).item()}, iters)


        depth = (depth - torch.min(depth))
        depth = depth / torch.max(depth)


        alpha = res[0][3:4].detach()
        alpha[alpha>1] = 1
        alpha[alpha<0] = 0

        swriter.add_image('vis/rendered', res[0][0:3], iters)
        swriter.add_image('vis/alpha', alpha, iters)
        swriter.add_image('vis/GT', target[0][0:3], iters)
        swriter.add_image('vis/GT_alpha', target[0][3:4], iters)
        swriter.add_image('vis/depth', depth[0], iters)
        # swriter.add_image('vis/ROI', target[0][4:5], iters)
        # if model.dataset is not None:
        #     swriter.add_image('vis/RGB', rgb[0], iters)



        return l.item()

    return Engine(_update)

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        swriter
):
    log_period = cfg.SOLVER.LOG_PERIOD
    log_period = 1
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("rendering_model.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, swriter = swriter )
   
    checkpointer = ModelCheckpoint(output_dir, 'nr', n_saved=10, require_empty=False)
    timer = Timer(average=True)

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,'optimizer': optimizer})
    trainer.add_event_handler(Events.ITERATION_COMPLETED, checkpointer, {'model': model,'optimizer': optimizer})


    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f} Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss'], scheduler.get_lr()[0]))
    

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()



    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)
