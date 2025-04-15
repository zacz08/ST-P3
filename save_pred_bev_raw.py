from argparse import ArgumentParser
from PIL import Image
import torch
import torch.utils.data
import numpy as np
import torchvision
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import matplotlib
from matplotlib import pyplot as plt
import pathlib
import datetime
import cv2
import os
import json

from stp3.datas.NuscenesData import FuturePredictionDataset
from stp3.trainer import TrainingModule
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.network import preprocess_batch, NormalizeInverse
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import make_contour

def mk_save_dir():
    now = datetime.datetime.now()
    string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
    save_path = pathlib.Path('imgs') / string
    save_path.mkdir(parents=True, exist_ok=False)
    return save_path

def eval(checkpoint_path, dataroot, mode, data_split):
    # save_folder_pred = mk_save_dir()
    folder_name = data_split
    save_folder_pred = os.path.join(dataroot, 'bev_pred_stp3_mask', folder_name)
    if not os.path.exists(save_folder_pred):
            os.makedirs(save_folder_pred)

    if mode == 'return_bev':
        save_folder_bevfeat = os.path.join(dataroot, 'bev_feat_raw_stp3', folder_name)
        if not os.path.exists(save_folder_bevfeat):
            os.makedirs(save_folder_bevfeat)

    bev_seg_gt_folder = os.path.join(dataroot, 'bev_seg_gt_stp3_mask', folder_name)
    json_name = 'prompt_stp3_mask_' + folder_name + '.json'
    json_path = os.path.join(dataroot, json_name)

    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)

    trainer.cfg.N_FUTURE_FRAMES = 1
    
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.PLANNING.ENABLED = False
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.LIFT.GT_DEPTH = False
    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.MAP_FOLDER = dataroot

    dataroot = cfg.DATASET.DATAROOT
    nworkers = cfg.N_WORKERS
    if 'mini' in args.data_split:
        nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
    else:
        nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)

    if 'train' in args.data_split:
        data_index = 0
    elif 'val' in args.data_split:
        data_index = 1
    elif 'test' in args.data_split:
        data_index = 2
    else:
        raise ValueError(f"Unexpected data_split value: {args.data_split}")
    
    valdata = FuturePredictionDataset(nusc, data_index, cfg)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False
    )

    n_classes = len(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
    hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
    metric_vehicle_val = IntersectionOverUnion(n_classes).to(device)
    future_second = int(cfg.N_FUTURE_FRAMES / 2)

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        metric_pedestrian_val = IntersectionOverUnion(n_classes).to(device)

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        metric_hdmap_val = []
        for i in range(len(hdmap_class)):
            metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1).to(device))

    if cfg.INSTANCE_SEG.ENABLED:
        metric_panoptic_val = PanopticMetric(n_classes=n_classes).to(device)

    # if cfg.PLANNING.ENABLED:
    #     metric_planning_val = []
    #     for i in range(future_second):
    #         metric_planning_val.append(PlanningMetric(cfg, 2*(i+1)).to(device))

    with open(json_path, 'w') as json_file:
        for index, batch in enumerate(tqdm(valloader)):
            preprocess_batch(batch, device)
            image = batch['image']
            intrinsics = batch['intrinsics']
            extrinsics = batch['extrinsics']
            future_egomotion = batch['future_egomotion']
            command = batch['command']
            trajs = batch['sample_trajectory']
            target_points = batch['target_point']
            bev_token = batch['bev_token'][2][0]
            B = len(image)
            labels = trainer.prepare_future_labels(batch)

            with torch.no_grad():
                output, bev_feat = model(
                    image, intrinsics, extrinsics, future_egomotion, mode
                )
            bev_feat_name = f"{index:05d}_bev_feat_{bev_token}.pt"
            save_path = os.path.join(save_folder_bevfeat, bev_feat_name)
            torch.save(bev_feat, save_path)

            n_present = model.receptive_field

            data_format = 'mask'
            save_bev(output, labels, batch, n_present, index, bev_token, save_folder_pred, format=data_format)

            # write data to json file
            gt_save_path = get_seg_map_name_by_sample_token(
                bev_seg_gt_folder,
                bev_token)
            assert gt_save_path is not None, f"Can't find bev gt image with sample_token: {bev_token}"
            pred_seg_map_path = f"{index:05d}_bev_pred_{bev_token}.jpg"
            
            if data_format == 'mask':
                pred_seg_map_path = pred_seg_map_path.replace('.jpg', '.npy')
                # gt_save_path = gt_save_path.replace('.jpg', '.npy')

            data = {
                "bev_feat": bev_feat_name,
                "pred_map": pred_seg_map_path,
                "bev_map_gt": gt_save_path
            }

            # 使用 json.dump 写入文件并换行
            json.dump(data, json_file)
            json_file.write('\n')  # 每行一个JSON对象


def save_bev(output, labels, batch, n_present, frame, bev_token, save_path_pred, format='image', visulisize=False):
    
    assert format in ['image', 'mask'], 'Invalid format!'
    
    hdmap = output['hdmap'].detach()
    segmentation = output['segmentation'][:, n_present - 1].detach()
    pedestrian = output['pedestrian'][:, n_present - 1].detach()

    if format == 'image':
        fig = plt.figure(1, figsize=(10, 10))

    # background: black
    showing = torch.zeros((200, 200, 3)).numpy()
    showing[:, :] = np.array([0 / 255, 0 / 255, 0 / 255])

    # drivable area
    area = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([31 / 255, 119 / 255, 180 / 255])
    if format == 'mask':
        drivable_mask = torch.zeros((200, 200)).numpy()
        drivable_mask[hdmap_index] = 1

    # lane
    area = torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([255 / 255, 127 / 255, 14 / 255])
    if format == 'mask':
        lane_mask = torch.zeros((200, 200)).numpy()
        lane_mask[hdmap_index] = 1

    # vehicle semantic
    semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
    semantic_index = semantic_seg > 0
    showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])
    if format == 'mask':
        vehicle_mask = torch.zeros((200, 200)).numpy()
        vehicle_mask[semantic_index] = 1

    # pedestrian semantic
    pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
    pedestrian_index = pedestrian_seg > 0
    showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])
    if format == 'mask':
        pedestrian_mask = torch.zeros((200, 200)).numpy()
        pedestrian_mask[pedestrian_index] = 1

    if format == 'mask':
        combined_mask = np.stack([drivable_mask, lane_mask, vehicle_mask, pedestrian_mask], axis=0)
        combined_mask = np.rot90(combined_mask, k=2, axes=(1, 2))
        
        save_path_pred = os.path.join(save_path_pred, f'{frame:05d}_bev_pred_{bev_token}.jpg')
        np.save(save_path_pred.replace('.jpg', '.npy'), combined_mask)

        if visulisize:
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(combined_mask[0], cmap='gray')
            plt.axis('off')
            plt.subplot(2, 2, 2)  
            plt.imshow(combined_mask[1], cmap='gray')
            plt.axis('off')
            plt.subplot(2, 2, 3) 
            plt.imshow(combined_mask[2], cmap='gray')
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.imshow(combined_mask[3], cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path_pred)

    elif format == 'image':

        plt.imshow(showing)
        plt.axis('off')

        # draw the ego vehicle
        bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
        dx = np.array([0.5, 0.5])
        # w, h = 1.85, 4.084
        w, h = 1.68, 3.45
        pts = np.array([
            [-h / 2. + 0.5, w / 2.],
            [h / 2. + 0.5, w / 2.],
            [h / 2. + 0.5, -w / 2.],
            [-h / 2. + 0.5, -w / 2.],
        ])
        pts = (pts - bx) / dx
        pts[:, [0, 1]] = pts[:, [1, 0]]
        plt.fill(pts[:, 0], pts[:, 1], 'w')

        fig.tight_layout(pad=0)
        fig.canvas.draw()  # render the plot
        
        # Convert matplotlib image to NumPy array
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # get image data
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))   # add channel dimension

        # rotate image 180 degree
        flipped_img = np.flipud(np.fliplr(img))
        plt.close()

        save_path_pred = os.path.join(save_path_pred, f'{frame:05d}_bev_pred_{bev_token}.jpg')
        cv2.imwrite(save_path_pred, cv2.cvtColor(cv2.resize(flipped_img, (512, 512)), cv2.COLOR_RGB2BGR))
    

def save(output, labels, batch, n_present, frame, save_path):
    hdmap = output['hdmap'].detach()
    segmentation = output['segmentation'][:, n_present - 1].detach()
    pedestrian = output['pedestrian'][:, n_present - 1].detach()
    gt_trajs = labels['gt_trajectory']
    images = batch['image']

    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )

    val_w = 2.99
    val_h = 2.99 * (224. / 480.)
    plt.figure(1, figsize=(4*val_w,2*val_h))
    width_ratios = (val_w,val_w,val_w,val_w)
    gs = matplotlib.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    plt.subplot(gs[0, 0])
    plt.annotate('FRONT LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,0].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 1])
    plt.annotate('FRONT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,1].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 2])
    plt.annotate('FRONT RIGHT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,2].cpu()))
    plt.axis('off')

    plt.subplot(gs[1, 0])
    plt.annotate('BACK LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0,n_present-1,3].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[1, 1])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 4].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[1, 2])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 5].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[:, 3])
    showing = torch.zeros((200, 200, 3)).numpy()
    showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255])

    # drivable
    area = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # lane
    area = torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])

    # semantic
    semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
    semantic_index = semantic_seg > 0
    showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])

    pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
    pedestrian_index = pedestrian_seg > 0
    showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])

    plt.imshow(make_contour(showing))
    plt.axis('off')

    bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    dx = np.array([0.5, 0.5])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    plt.xlim((200, 0))
    plt.ylim((0, 200))
    gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
    gt_trajs = (gt_trajs[0, :, :2].cpu().numpy() - bx) / dx
    plt.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=3.0)

    plt.savefig(save_path / ('%04d.png' % frame))
    plt.close()


def get_seg_map_name_by_sample_token(
        folder_path, 
        sample_token: str) -> str:
        
    for filename in os.listdir(folder_path):
        # get token by image name
        if filename.endswith(".jpg") or filename.endswith(".npy"):
            current_token = filename[len("00262_bev_gt_"):-len(".jpg")]
            if current_token == sample_token:
                return filename
    
    # return none if can't find the image
    return None

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', required=True, type=str)
    parser.add_argument('--data_split', 
                        required=True, 
                        type=str, 
                        choices=['train', 'val', 'test', 'mini_train', 'mini_val'],
                        help='Dataset split: must be one of {train, val, test, mini_train, mini_val}')

    args = parser.parse_args()

    eval(checkpoint_path=args.checkpoint, 
         dataroot=args.dataroot,
         mode='return_bev',
         data_split=args.data_split)
