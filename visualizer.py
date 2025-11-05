"""
BEV Visualization Tools
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Arrow
from typing import List, Dict, Tuple
import cv2
import os


class BEVVisualizer:
    """
    Visualizer for BEV (Bird's Eye View) space.
    """
    
    def __init__(self, config: dict):
        """
        Initialize BEV visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.bev_range = config['visualization']['bev_range']  # [x_min, x_max, y_min, y_max]
        self.grid_size = config['visualization']['grid_size']
        self.save_dir = config['visualization']['save_dir']
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Class colors (RGB)
        self.class_colors = {
            'car': (0, 0, 255),  # Blue
            'truck': (255, 0, 0),  # Red
            'bus': (255, 165, 0),  # Orange
            'trailer': (128, 0, 128),  # Purple
            'construction_vehicle': (165, 42, 42),  # Brown
            'pedestrian': (0, 255, 0),  # Green
            'motorcycle': (255, 255, 0),  # Yellow
            'bicycle': (0, 255, 255),  # Cyan
            'traffic_cone': (255, 192, 203),  # Pink
            'barrier': (128, 128, 128)  # Gray
        }
    
    def plot_bev(self, 
                 gt_objects: List[Dict] = None,
                 pred_objects: List[Dict] = None,
                 title: str = "BEV Visualization",
                 save_path: str = None,
                 show: bool = False) -> np.ndarray:
        """
        Plot objects in BEV space.
        
        Args:
            gt_objects: List of ground truth objects with BEV coordinates
            pred_objects: List of predicted objects with BEV coordinates
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Figure as numpy array
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set axis limits (rotated 90 degrees CCW: X becomes Y, Y becomes -X)
        x_min, x_max, y_min, y_max = self.bev_range
        ax.set_xlim(-y_max, -y_min)  # Y-axis becomes horizontal (flipped)
        ax.set_ylim(x_min, x_max)    # X-axis becomes vertical (forward is up)
        
        # Draw grid
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('Y (meters) - Left/Right')
        ax.set_ylabel('X (meters) - Forward')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        # Draw ego vehicle at origin (rotated 90 degrees CCW)
        ego_width = 2.0
        ego_length = 4.5
        ego_rect = Rectangle((-ego_width/2, -ego_length/2), ego_width, ego_length,
                            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5)
        ax.add_patch(ego_rect)
        
        # Plot ground truth objects
        if gt_objects:
            for obj in gt_objects:
                self._draw_bev_box(ax, obj, color='green', linestyle='-', label='GT', alpha=0.5)
        
        # Plot predicted objects
        if pred_objects:
            for obj in pred_objects:
                self._draw_bev_box(ax, obj, color='red', linestyle='--', label='Pred', alpha=0.7)
        
        # Add legend
        if gt_objects or pred_objects:
            handles = []
            labels = []
            if gt_objects:
                handles.append(plt.Line2D([0], [0], color='green', linewidth=2, linestyle='-'))
                labels.append('Ground Truth')
            if pred_objects:
                handles.append(plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--'))
                labels.append('Prediction')
            ax.legend(handles, labels, loc='upper right')
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved BEV visualization to {save_path}")
        
        # Convert figure to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return img
    
    def _draw_bev_box(self, ax, obj: Dict, color: str, linestyle: str, label: str, alpha: float = 0.7):
        """
        Draw a bounding box in BEV space.
        
        Args:
            ax: Matplotlib axis
            obj: Object dictionary with x, y, yaw, width, length
            color: Box color
            linestyle: Line style
            label: Label (GT or Pred)
            alpha: Transparency
        """
        x = obj['x']
        y = obj['y']
        yaw = obj['yaw']
        width = obj.get('width', 2.0)
        length = obj.get('length', 4.0)
        obj_class = obj.get('class', 'unknown')
        
        # Get class-specific color if available
        if obj_class in self.class_colors:
            rgb = self.class_colors[obj_class]
            color = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
        
        # Compute box corners
        corners = self._get_box_corners(x, y, yaw, width, length)
        
        # Rotate corners 90 degrees CCW for visualization: (x, y) -> (-y, x)
        corners_rotated = np.column_stack((-corners[:, 1], corners[:, 0]))
        
        # Draw box
        box = patches.Polygon(corners_rotated, closed=True, 
                             edgecolor=color, facecolor=color,
                             linestyle=linestyle, linewidth=2, alpha=alpha)
        ax.add_patch(box)
        
        # Draw orientation arrow (rotated 90 degrees CCW)
        arrow_length = length * 0.5
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        # Rotate arrow: (dx, dy) -> (-dy, dx)
        dx_rot = -dy
        dy_rot = dx
        # Rotate position: (x, y) -> (-y, x)
        x_rot = -y
        y_rot = x
        ax.arrow(x_rot, y_rot, dx_rot, dy_rot, head_width=0.5, head_length=0.3, 
                fc=color, ec=color, alpha=alpha)
        
        # Add text label (rotated position)
        ax.text(x_rot, y_rot, obj_class, fontsize=8, color='white',
               ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    def _get_box_corners(self, x: float, y: float, yaw: float, 
                        width: float, length: float) -> np.ndarray:
        """
        Get corners of a bounding box in BEV.
        
        Args:
            x, y: Center position
            yaw: Rotation angle
            width, length: Box dimensions
            
        Returns:
            Array of corner coordinates (4, 2)
        """
        # Box corners in local frame
        corners_local = np.array([
            [length/2, width/2],
            [length/2, -width/2],
            [-length/2, -width/2],
            [-length/2, width/2]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        
        # Transform to global frame
        corners_global = corners_local @ R.T + np.array([x, y])
        
        return corners_global
    
    def visualize_detections_2d(self, 
                               image: np.ndarray,
                               detections: List[Dict],
                               gt_boxes: List[Dict] = None,
                               save_path: str = None,
                               show: bool = False) -> np.ndarray:
        """
        Visualize 2D detections on image.
        
        Args:
            image: Input image
            detections: List of detections
            gt_boxes: List of ground truth boxes
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Image with drawn detections
        """
        img_vis = image.copy()
        
        # Draw ground truth boxes in green
        if gt_boxes:
            for box in gt_boxes:
                bbox = box['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"GT: {box.get('class', 'unknown')}"
                cv2.putText(img_vis, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw detections in red
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(img_vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
            print(f"Saved 2D visualization to {save_path}")
        
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(img_vis)
            plt.axis('off')
            plt.title('2D Object Detections')
            plt.tight_layout()
            plt.show()
        
        return img_vis
    
    def create_comparison_plot(self,
                              image: np.ndarray,
                              detections: List[Dict],
                              gt_bev: List[Dict],
                              pred_bev: List[Dict],
                              sample_token: str,
                              save_dir: str = None) -> None:
        """
        Create a side-by-side comparison plot showing 2D detections and BEV.
        
        Args:
            image: Input image
            detections: 2D detections
            gt_bev: Ground truth BEV objects
            pred_bev: Predicted BEV objects
            sample_token: Sample identifier
            save_dir: Directory to save plot
        """
        fig = plt.figure(figsize=(20, 8))
        
        # 2D detections
        ax1 = fig.add_subplot(1, 2, 1)
        img_with_dets = self.visualize_detections_2d(image, detections)
        ax1.imshow(img_with_dets)
        ax1.set_title('2D Object Detections')
        ax1.axis('off')
        
        # BEV visualization
        ax2 = fig.add_subplot(1, 2, 2)
        
        # Set axis limits (rotated 90 degrees CCW)
        x_min, x_max, y_min, y_max = self.bev_range
        ax2.set_xlim(-y_max, -y_min)  # Y-axis becomes horizontal (flipped)
        ax2.set_ylim(x_min, x_max)    # X-axis becomes vertical (forward is up)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_xlabel('Y (meters) - Left/Right')
        ax2.set_ylabel('X (meters) - Forward')
        ax2.set_title('BEV Space')
        ax2.set_aspect('equal')
        
        # Draw ego vehicle (rotated 90 degrees CCW)
        ego_width = 2.0
        ego_length = 4.5
        ego_rect = Rectangle((-ego_width/2, -ego_length/2), ego_width, ego_length,
                            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5)
        ax2.add_patch(ego_rect)
        
        # Plot ground truth and predictions
        if gt_bev:
            for obj in gt_bev:
                self._draw_bev_box(ax2, obj, color='green', linestyle='-', label='GT', alpha=0.5)
        
        if pred_bev:
            for obj in pred_bev:
                self._draw_bev_box(ax2, obj, color='red', linestyle='--', label='Pred', alpha=0.7)
        
        # Add legend
        handles = [
            plt.Line2D([0], [0], color='green', linewidth=2, linestyle='-'),
            plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--')
        ]
        labels = ['Ground Truth', 'Prediction']
        ax2.legend(handles, labels, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{sample_token}_comparison.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        plt.close(fig)


def compute_bev_metrics(gt_objects: List[Dict], pred_objects: List[Dict]) -> Dict[str, float]:
    """
    Compute evaluation metrics for BEV predictions.
    
    Args:
        gt_objects: Ground truth BEV objects
        pred_objects: Predicted BEV objects
        
    Returns:
        Dictionary of metrics
    """
    if not gt_objects or not pred_objects:
        return {
            'position_mse': float('inf'),
            'position_mae': float('inf'),
            'orientation_error': float('inf'),
            'size_error': float('inf')
        }
    
    # Match predictions to ground truth (simplified: assume same order)
    min_len = min(len(gt_objects), len(pred_objects))
    
    position_errors = []
    orientation_errors = []
    size_errors = []
    
    for i in range(min_len):
        gt = gt_objects[i]
        pred = pred_objects[i]
        
        # Position error (Euclidean distance)
        pos_error = np.sqrt((gt['x'] - pred['x'])**2 + (gt['y'] - pred['y'])**2)
        position_errors.append(pos_error)
        
        # Orientation error (angle difference)
        orient_error = abs(gt['yaw'] - pred['yaw'])
        # Normalize to [-pi, pi]
        orient_error = (orient_error + np.pi) % (2 * np.pi) - np.pi
        orientation_errors.append(abs(orient_error))
        
        # Size error (average difference in dimensions)
        size_error = (abs(gt['width'] - pred['width']) + 
                     abs(gt['length'] - pred['length']) + 
                     abs(gt['height'] - pred['height'])) / 3
        size_errors.append(size_error)
    
    metrics = {
        'position_mse': np.mean(np.array(position_errors)**2),
        'position_mae': np.mean(position_errors),
        'orientation_error': np.mean(orientation_errors),
        'size_error': np.mean(size_errors),
        'num_matched': min_len
    }
    
    return metrics
