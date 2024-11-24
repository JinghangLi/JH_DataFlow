from typing import Dict, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_3d_bbox_projection(projected_points: np.ndarray, image: np.ndarray, color: np.ndarray, label: str, score) -> np.ndarray:
    """
    在给定的图像上绘制3D包围框的投影。

    参数:
    - projected_points: 包围框八个角点的投影坐标，大小为 (8, 2)
    - image: 要绘制的图像，类型为 np.ndarray

    返回:
    - 绘制了3D包围框投影的图像
    """
    # 定义边的连接关系
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # x0 平面四条边
        (4, 5), (5, 6), (6, 7), (7, 4),  # x1 平面四条边
        (0, 4), (1, 5), (2, 6), (3, 7)   # 连接 x0 和 x1 平面的四条边
    ]
    projected_points = projected_points.astype(int)
    # 画出八个角点
    for point in projected_points:
        cv2.circle(image, (int(point[0]), int(point[1])), 4, color, -1)  # 使用蓝色圆点表示角点 (BGR 格式)

    # 画出对应的连线
    for edge in edges:
        point1 = tuple(projected_points[edge[0]])
        point2 = tuple(projected_points[edge[1]])
        cv2.line(image, point1, point2, color, 2)  # 使用红色线连接对应的点 (BGR 格式)
    
    cv2.putText(image, f'{label}_{score:.2f}', (int(projected_points[3, 0])+5, int(projected_points[3, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 
    
    return image


def plot_mached_3d_bbox_projection(projected_points: np.ndarray, image: np.ndarray, color: np.ndarray, iou) -> np.ndarray:
    """
    在给定的图像上绘制3D包围框的投影。

    参数:
    - projected_points: 包围框八个角点的投影坐标，大小为 (8, 2)
    - image: 要绘制的图像，类型为 np.ndarray

    返回:
    - 绘制了3D包围框投影的图像
    """
    # 定义边的连接关系
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # x0 平面四条边
        (4, 5), (5, 6), (6, 7), (7, 4),  # x1 平面四条边
        (0, 4), (1, 5), (2, 6), (3, 7)   # 连接 x0 和 x1 平面的四条边
    ]
    projected_points = projected_points.astype(int)
    # 画出八个角点
    for point in projected_points:
        cv2.circle(image, (int(point[0]), int(point[1])), 4, color, 1)

    # 画出对应的连线
    for edge in edges:
        point1 = tuple(projected_points[edge[0]])
        point2 = tuple(projected_points[edge[1]])
        cv2.line(image, point1, point2, color, 1)  # 使用红色线连接对应的点 (BGR 格式)
    
    cv2.putText(image, f'mached_{iou:.1f}', (int(projected_points[3, 0])-20, int(projected_points[3, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 
    
    return image


def plot_pseudo_bbox_on_image_and_bev(raw_image: np.ndarray, 
                                        imgfov_pts_2d: np.ndarray,
                                        corner_box3d_lidar: np.ndarray,
                                        point_cloud: np.ndarray, 
                                        pseudo_meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    
    '''
    Draw pseudo label on image
    return: image_pseudo and bev_image
    '''
    box_label = pseudo_meta['labels_3d']
    box_color = pseudo_meta['color_3d']
    box_score = pseudo_meta['score_3d']
    good_class = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']
    
    bev_plot_xlim = (-1, 30)
    bev_plot_ylim = (-15, 15)
    
    # prepare image_pseudo
    image_pseudo = raw_image
    
    # prepare bev view, create the figure and axis
    fig, ax = plt.subplots(figsize=(32, 25), dpi=150)  # Increase DPI for better clarity
    ax.scatter(x=point_cloud[:, 0], y=point_cloud[:, 1], s=100, c=point_cloud[:, 3], cmap='viridis', alpha=0.5, marker='.')
    # Draw x and y axis arrows from origin
    ax.arrow(0, 0, 5, 0, head_width=0.5, head_length=0.5, fc='r', ec='r', linewidth=3)
    ax.arrow(0, 0, 0, 5, head_width=0.5, head_length=0.5, fc='b', ec='b', linewidth=3)

    # Draw the 3D bounding boxes
    for imgfov_2d_corner, lidar_3d_corner, color, label, score in zip(imgfov_pts_2d, corner_box3d_lidar, box_color, box_label, box_score): 
        bool_in_image = np.any((imgfov_2d_corner[:, 0] > 0) & (imgfov_2d_corner[:, 0] < 640) & (imgfov_2d_corner[:, 1] > 0) & (imgfov_2d_corner[:, 1] < 480))
        bool_term = label in good_class and score > 0.2 and bool_in_image
        if bool_term:
            image_pseudo = plot_3d_bbox_projection(
                projected_points=imgfov_2d_corner, 
                image=image_pseudo, 
                color=color, 
                label=label, 
                score=score)

            # draw bev
            # Extract bounding box corner coordinates          
            color_normalized = tuple([c / 255.0 for c in color])
            # get upper 4 points xy cords of the 3d box
            x_coords = [lidar_3d_corner[i][0] for i in [1, 2, 6, 5, 1]]
            y_coords = [lidar_3d_corner[i][1] for i in [1, 2, 6, 5, 1]]
            # check if all the points are in the bev_plot_xlim and bev_plot_ylim
            if np.any((bev_plot_xlim[0] < np.array(x_coords)) & (np.array(x_coords) < bev_plot_xlim[1])) and \
                np.any((bev_plot_ylim[0] < np.array(y_coords)) & (np.array(y_coords) < bev_plot_ylim[1])):
    
                ax.plot(x_coords, y_coords, color=color_normalized, linewidth=5)
                # Place text label parallel to the y-axis at the bottom edge of the bounding box
                text_x = (min(x_coords) + max(x_coords)) / 2 
                text_y = max(y_coords)
                ax.text(text_x, text_y, f'{label}_{score:.2f}', color=color_normalized, fontsize=50, verticalalignment='top', rotation=-90)
            
    # Convert the BEV plot to a numpy array
    # Set axis limits, labels, and title
    ax.set_xlim(bev_plot_xlim)
    ax.set_ylim(bev_plot_ylim)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.grid(True)
    fig.tight_layout()
    fig.canvas.draw()
    # Convert figure to PIL image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    pil_img = Image.fromarray(img)
    # Rotate image 90 degrees counterclockwise
    rotated_img = pil_img.rotate(90, expand=True)
    # Resize image to height of 480 while maintaining aspect ratio
    aspect_ratio = rotated_img.width / rotated_img.height
    new_width = int(480 * aspect_ratio)
    bev_view = np.array(rotated_img.resize((new_width, 480)))
    # Close the figure to free memory
    plt.close(fig)
            
    return image_pseudo, bev_view 


def plot_anno_bbox_on_image(image: np.ndarray, box: List[float], class_name: str) -> np.ndarray:
    '''
    Draws a rectangle on the image.
    box: [x1, y1, x2, y2]
    '''
    max_y, max_x, _ = image.shape
    x1, y1, x2, y2 = [int(itm) for itm in box]
    
    x1_bounds = max(0, min(x1, max_x))
    y1_bounds = max(0, min(y1, max_y))
    x2_bounds = max(0, min(x2, max_x))
    y2_bounds = max(0, min(y2, max_y))
    
    # Draw rectangle on the image
    image_with_rectangle = image.copy()
    color = (0, 255, 0)  # Green color for the rectangle
    thickness = 1  # Thickness of the rectangle border
    cv2.rectangle(image_with_rectangle, (x1_bounds, y1_bounds), (x2_bounds, y2_bounds), color, thickness)
    cv2.putText(image_with_rectangle, class_name, (x1_bounds, y1_bounds), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 
    
    return image_with_rectangle


def plot_bev_projection(point_cloud, lidar_3d_corner, label, score, color):
    """
    Draws a pseudo bounding box on the given point cloud data and returns the resulting image.

    Parameters:
    - point_cloud: numpy.ndarray, point cloud data (N, 4), where columns represent x, y, z, and intensity.
    - lidar_3d_corner: list of lists, containing the corner points of the 3D bounding box.
    - label: str, label for the bounding box.
    - score: float, confidence score for the bounding box.
    - color: tuple, RGB color for the bounding box.

    Returns:
    - numpy.ndarray, the resulting image with the bounding box and label drawn.
    """
    # Convert color to a valid format (normalize RGB values to [0, 1])
    color_normalized = tuple([c / 255.0 for c in color])

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(32, 25), dpi=150)  # Increase DPI for better clarity
    ax.scatter(x=point_cloud[:, 0], y=point_cloud[:, 1], s=100, c=point_cloud[:, 3], cmap='viridis', alpha=0.5, marker='.')

    # Extract bounding box corner coordinates
    x_coords = [lidar_3d_corner[i][0] for i in [1, 2, 6, 5, 1]]
    y_coords = [lidar_3d_corner[i][1] for i in [1, 2, 6, 5, 1]]
    ax.plot(x_coords, y_coords, color=color_normalized, linewidth=5)

    # Place text label parallel to the y-axis at the bottom edge of the bounding box
    text_x = (min(x_coords) + max(x_coords)) / 2 
    text_y = max(y_coords)
    ax.text(text_x, text_y, f'{label}_{score:.2f}', color=color_normalized, fontsize=50, verticalalignment='top', rotation=-90)

    # Set axis limits, labels, and title
    ax.set_xlim(-1, 25)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('BEV')
    ax.grid(True)
    fig.tight_layout()
    fig.canvas.draw()

    # Convert figure to PIL image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    pil_img = Image.fromarray(img)

    # Rotate image 90 degrees counterclockwise
    rotated_img = pil_img.rotate(90, expand=True)

    # Resize image to height of 480 while maintaining aspect ratio
    aspect_ratio = rotated_img.width / rotated_img.height
    new_width = int(480 * aspect_ratio)
    resized_img = np.array(rotated_img.resize((new_width, 480)))
    
    # Close the figure to free memory
    plt.close(fig)
    return resized_img

# Example usage
# result_img = draw_pseudo_bbox_on_image(point_cloud, lidar_3d_corner, label, score, (255, 158, 0))
# Note: Replace `point_cloud`, `lidar_3d_corner`, `label`, and `score` with actual values when using the function.