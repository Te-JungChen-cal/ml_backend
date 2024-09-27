import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

class ml_yolo:
    def __init__(self):
        """load the yolo model"""
        model_variant = 'yolov8l.pt' 

        # Load the YOLOv8 model
        try:
            self.model = YOLO(model_variant)
            print(f"YOLOv8 '{model_variant}' model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")

    def detect_inventory(self, image_path, confidence_threshold=0.3, iou_threshold=0.5):
        """
        Detects inventory items in the given image using YOLOv8 and returns a DataFrame.

        Args:
            image_path (str): Path to the image file.
            confidence_threshold (float): Minimum confidence score for a detection to be considered valid.
            iou_threshold (float): Intersection over union threshold for non-max suppression.

        Returns:
            tuple: A DataFrame containing detected items and their confidence scores, 
                   and the raw results from the YOLO model.
        """

        # Verify that the image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at path: {image_path}")
            return pd.DataFrame(), None

        try:
            # Perform prediction using the YOLO model
            results = self.model.predict(
                source=image_path,
                conf=confidence_threshold,  # Set confidence threshold
                iou=iou_threshold,          # Set IoU threshold
                save=False,                 # Do not save predictions by default
                verbose=False               # Suppress verbose output
            )
        except Exception as e:
            print(f"Error during prediction: {e}")
            return pd.DataFrame(), None

        first_result = results[0] if results else None
        detections = first_result.boxes if first_result else None

        if detections is None or len(detections) == 0:
            print("No items detected in the image.")
            return pd.DataFrame(), results

        # Initialize an inventory list to store detected items
        inventory = []
        for box in detections:
            cls_id = int(box.cls)  # Class ID
            confidence = box.conf.item()  # Confidence score
            class_name = self.model.names[cls_id]  # Get class name from model's class names
            if confidence < confidence_threshold:
                continue # Skip items below the confidence threshold
            inventory.append([class_name, confidence])

        # Create a DataFrame to store the inventory
        inventory_df = pd.DataFrame(inventory, columns=['Item', 'Confidence'])
        inventory_df['Count'] = 1
        inventory_df = inventory_df.groupby('Item').agg({'Count': 'sum', 'Confidence': 'mean'}).reset_index()
        return inventory_df, results

    def save_results(self, image_path, results):
        """
        Save the image with bounding boxes and labels using matplotlib.

        Args:
            image_path (str): Path to the image file.
            results: The results from the YOLO model containing detections.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image with OpenCV: {image_path}")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')  # Hide axis

        first_result = results[0] if results else None
        detections = first_result.boxes if first_result else None

        if detections is None or len(detections) == 0:
            print("No detections to display.")
            return

        # Loop through each detection to draw bounding boxes and labels
        for box in detections:
            # Get class ID of the detected item
            cls_id = int(box.cls) 

            # Get confidence score of the detection
            confidence = box.conf.item()  

            # Get class name using class ID
            class_name = self.model.names[cls_id]  

            # Get the bounding box coordinates
            xyxy = box.xyxy.tolist() 

            # Check if the bounding box is a list and convert it if needed
            if isinstance(xyxy[0], list):
                xyxy = xyxy[0]

            # Unpack bounding box coordinates and convert to int
            x1, y1, x2, y2 = map(int, xyxy)  

            # Create label with class name and confidence
            label = f"{class_name} {confidence:.2f}"  

            # Add a rectangle for the bounding box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            fill=False, edgecolor='red', linewidth=2))
            # Add label text above the bounding box
            plt.text(x1, y1 - 10, label, color='red', fontsize=12,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def process_image(self, image_path):
        """
        Process the image to detect inventory and save results.

        Args:
            image_path (str): Path to the image file.

        Returns:
            DataFrame: DataFrame containing detected inventory items.
        """
        # Call detection method
        inventory_df, results = self.detect_inventory(image_path)
        
        # Show the image with detected bounding boxes
        if results:
            # Save the results with bounding boxes
            self.save_results(image_path, results)
            return inventory_df
        else:
            return None
