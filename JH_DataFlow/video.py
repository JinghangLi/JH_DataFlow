import cv2
import os
import glob
from pathlib import Path

def images_to_video(img_path_list, save_path, fps=20):

    text_params = dict(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),
        thickness=2,
    )
    
    assert len(img_path_list) > 0, "No images found in the list."
    
    first_image = cv2.imread(img_path_list[0])
    height, width, _ = first_image.shape
    
    # For compare two images 
    # height = height * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择编码格式
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for img_path in img_path_list:
        img = cv2.imread(img_path)
        cv2.putText(img, f"{Path(img_path).stem}", (20, 60), **text_params)
        
        # For compare two images 
        # relative_path = Path(img_path).relative_to('/home/jhang/workspace/Dataset_tools/ettc_prepare/result/ettc_dsec/nnETTC_v0.0_finetune')
        # img_compare_path = (Path('result/ettc_dsec/nnETTC_v2.0') / relative_path).absolute()
        # img_compare = cv2.imread(img_compare_path.as_posix())
        # img = cv2.vconcat([img, img_compare])
         
        video_writer.write(img)

    video_writer.release()
    print(f"Video saved at {save_path}")
