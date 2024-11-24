import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from .io import read_image



def vis_cuboid_anno(cuboid_anno: List[Dict], image: np.ndarray, save_path: str) -> None:
    img = image
    rectangle_color = (0, 0, 255)
    rectangle_color_2 = (0, 255, 0)
    text_color = (255, 255, 255)
    for itm in cuboid_anno:
        x1, y1, x2, y2 = [int(itm_) for itm_ in itm['box']]
        kp_x = int(itm['kp_u'])
        visbdr_l, visbdr_r = [int(itm_) for itm_ in itm['visbdr']]
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color=rectangle_color,
            thickness=4,
        )
        cv2.putText(
            img,
            f'{itm["trackid"]}_{itm["cls_name"]}_{itm["kp_index"]}_{itm["by_hand"]}',
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=text_color,
            thickness=2,
        )
        cv2.circle(img, (kp_x, y2), 5, text_color, -1)
        cv2.rectangle(
            img,
            (visbdr_l, y1),
            (visbdr_r, y2),
            color=rectangle_color_2,
            thickness=2,
        )
    cv2.imwrite(str(save_path), img)
    return


def vis_patches_single_asset(
    patch_list: List[str],
    save_dir: str,
    num_workers: int,
):
    ''' Visualize patches (multi-processing)
    Args:
        patch_list (List[str])
        save_dir (str)
    '''
    if num_workers <= 1:
        vis_patches(patch_list=patch_list, save_dir=save_dir)
    else:
        param_list = [{
            'patch_list': itm.tolist(),
            'save_dir': save_dir,
        } for itm in np.array_split(patch_list, num_workers)]
        with Pool(num_workers) as p:
            list(tqdm(p.imap(mp_vis_patches, param_list), desc='mp_vis_patches(multi-process)'))
    return


def mp_vis_patches(kwargs) -> None:
    """Multi-process function for calling vis_patches."""
    return vis_patches(**kwargs)


def vis_patches(
    patch_list: List[str],
    save_dir: str,
) -> None:
    ''' Visualize patches
    Args:
        patch_list (List[str])
        save_dir (str)
    '''
    rect_color_1 = (0, 255, 0)
    rect_color_2 = (0, 255, 255)
    text_color = (255, 255, 255)
    circle_color = (0, 0, 255)
    thickness = 2
    line_itv = 25
    text_params = dict(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=text_color,
        thickness=thickness,
    )
    for patch_path in tqdm(patch_list, desc='vis_patches'):
        anno_path = str(Path(patch_path)).replace('.png', '.npy')
        anno = np.load(anno_path, allow_pickle=True).item()
        image_shape = anno['meta']['image_shape']
        image = read_image(str(patch_path))
        largebox = [int(itm) for itm in anno['box_l']]
        box = [int(itm) for itm in anno['box']]
        kp_x = int(anno['kp_u'])
        timestamp = Path(patch_path).name.split('_')[1]
        ttc = anno['ttc']
        httc = anno['horizontal_ttc']
        x_Fego, y_Fego = anno['box3d'].flatten()[:2]
        y_speed_ego = np.array(anno['speed_ego_10x']).flatten()[1]
        ego_vel = anno['ego_vel']
        x_rel_speed_ego = np.linalg.norm(ego_vel) - anno['speed_ego_10x'][0]
        box_h = int(anno['box'][3] - anno['box'][1])
        box_w = int(anno['box'][2] - anno['box'][0])
        lbox_h = int(anno['box_l'][3] - anno['box_l'][1])
        lbox_w = int(anno['box_l'][2] - anno['box_l'][0])

        delta_t = 0.1
        s = 1 + delta_t / anno['ttc']
        s_h = 1 + delta_t / anno['horizontal_ttc']
        cv2.rectangle(image, largebox[:2], largebox[2:], rect_color_2, max(thickness - 1, 1))  # 脑补框
        cv2.rectangle(image, box[:2], box[2:], rect_color_1, thickness)  # 非脑补框一个面
        cv2.circle(image, (kp_x, box[3]), 10, circle_color, -1)  # 拐点
        vis_image = stick_patch_to_target_image(
            image=image,
            box_on_image=anno['box'],
            box_on_origin_image=anno['box_Fimage'],
            origin_image_shape=anno['meta']['image_shape'],
            target_image=np.zeros(image_shape, np.uint8),
        )
        cv2.putText(vis_image, f'{timestamp}', (30, 30), **text_params)
        s_list = []
        s_list += [f'{anno["trackid"]}_{anno["cls_name"]}']
        s_list += [f'ttc{ttc:.1f}_s{s:.2f}']
        s_list += [f'httc{httc:.1f}_sh{s_h:.2f}']
        s_list += [f'x{x_Fego:.1f}_vx{x_rel_speed_ego:.1f}']
        s_list += [f'y{y_Fego:.1f}_vy{y_speed_ego:.1f}']
        s_list += [f'box_h{box_h}_box_w{box_w}']
        s_list += [f'lbox_h{lbox_h}_lbox_w{lbox_w}']
        s_list += [
            f'box{anno["meta"]["scores"]["bboxes_score"]:.1f}'
            f'ord{anno["meta"]["scores"]["onroad_score"]:.1f}'
            f'cbd{anno["meta"]["scores"]["cuboid_score"]:.1f}'
            f'valid{anno["meta"]["scores"]["vote_score_valid_ratio"]:.1f}'
            f'occ{anno["meta"]["scores"]["occ_ratio"]:.1f}'
        ]
        for i_s, s in enumerate(s_list):
            cv2.putText(vis_image, s, (30, 60 + i_s * line_itv), **text_params)
        cv2.imwrite(str(Path(save_dir) / Path(patch_path).name), vis_image)
    return


def ffmpeg_images(
    image_dir: str,
    image_surfix: str,
    video_save_path: str,
) -> None:
    '''Generate a video from images
    Args:
        image_dir: str
        image_surfix: str
        video_save_path: str
    '''
    ffmpeg_cmd = 'ffmpeg -threads 0 -y -framerate 10 -pattern_type glob -i '
    ffmpeg_cmd += f"'{Path(image_dir)/ ('*.'+image_surfix)}' "
    ffmpeg_cmd += '-c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p '
    ffmpeg_cmd += video_save_path
    logging.info(ffmpeg_cmd)
    return os.system(ffmpeg_cmd)


def stick_patch_to_target_image(
    image: np.ndarray,
    box_on_image: np.ndarray,
    box_on_origin_image: np.ndarray,
    origin_image_shape: Tuple[int, int],
    target_image: np.ndarray,
) -> np.ndarray:
    '''' Stick patch to target_image
    Args:
        image(np.ndarray, [height, width, ...]): patch = image[box_on_image]
        box_on_image( np.ndarray, [4,]): patch = image[box_on_image]
        box_on_origin_image(np.ndarray, [4, ...]): it contains the position of the patch in original image
        origin_image_shape( Tuple[int, int]): [H, W]
        target_image( np.ndarray, [height_target, width_target, ...])
    Returns:
        target_image( np.ndarray, [height_target, width_target, ...])
    '''
    box = box_on_image
    box_Fimage = box_on_origin_image
    H, W = np.array(origin_image_shape).flatten()[:2]
    H_target, W_target = target_image.shape[0], target_image.shape[1]
    box_target = np.array([
        box_Fimage[0] - np.floor((W - W_target) / 2),
        box_Fimage[1] - np.floor((H - H_target) / 2),
        box_Fimage[2] - np.floor((W - W_target) / 2),
        box_Fimage[3] - np.floor((H - H_target) / 2),
    ]).astype(np.int64)
    err_msg = f'box_target: {box_target}'
    is_valid = (box_target[0] >= 0 and box_target[1] >= 0 and box_target[2] <= W_target and box_target[3] <= H_target)
    if not is_valid:
        logging.error(err_msg)
        return target_image
    x1, y1, x2, y2 = np.array(box).astype(np.int32).flatten()
    x1_, y1_, x2_, y2_ = np.array(box_target).astype(np.int32).flatten()
    h, w, _ = image[y1:y2, x1:x2, ...].shape
    target_image[y1_:y1_ + h, x1_:x1_ + w, ...] = image[y1:y2, x1:x2, ...]
    return target_image


def visualize_patches_and_generate_video(
    patch_list: List[str],
    save_dir: str,
    video_save_path: str,
    num_workers: int,
):
    vis_patches_single_asset(
        patch_list=patch_list,
        save_dir=save_dir,
        num_workers=num_workers,
    )
    ffmpeg_images(
        image_dir=save_dir,
        image_surfix='png',
        video_save_path=video_save_path,
    )
    return
