# -*- coding: utf-8 -*-
import os
import glob
import pickle
import numpy as np
import torch
from PIL import Image, ImageDraw
import pandas as pd

# Global Imports
from deepface import DeepFace
from deepface.modules import verification as dst

# --- Helper Functions for Image Conversion ---
def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Converts a torch tensor to a PIL Image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image to a torch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- Common Function for Building the Database ---
def _build_db_logic(directory_path, db_save_path, model_name, detector_backend, force_rebuild):
    if not os.path.isdir(directory_path):
        return (None, f"Error: Directory not found at {directory_path}")
    
    output_dir = os.path.dirname(db_save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not force_rebuild and os.path.exists(db_save_path):
        try:
            with open(db_save_path, 'rb') as f:
                all_embeddings_data = pickle.load(f)
            status = f"Loaded {len(all_embeddings_data)} faces from DB."
            return (all_embeddings_data, status)
        except Exception as e:
            print(f"Could not load DB file, will rebuild. Error: {e}")

    print(f"Generating new embeddings using model: {model_name}")
    all_embeddings_data = []
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
        image_paths.extend(glob.glob(os.path.join(directory_path, ext)))

    if not image_paths:
        return (None, f"No images found in {directory_path}.")

    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        try:
            img_representations = DeepFace.represent(
                img_path=img_path, model_name=model_name,
                detector_backend=detector_backend, enforce_detection=True
            )
            for rep_idx, rep_obj in enumerate(img_representations):
                embedding_data = {
                    'target_image': os.path.basename(img_path),
                    'image_path': img_path, 'embedding': rep_obj['embedding'],
                    'facial_area': rep_obj['facial_area'], 'face_index_in_image': rep_idx,
                    'model_name': model_name
                }
                all_embeddings_data.append(embedding_data)
        except Exception as e:
            print(f"  Skipping {os.path.basename(img_path)} due to error: {e}")

    if all_embeddings_data:
        with open(db_save_path, 'wb') as f:
            pickle.dump(all_embeddings_data, f)
        status = f"Saved {len(all_embeddings_data)} face embeddings to DB."
    else:
        status = "No faces were detected in any images (or a processing error occurred)."
        return (None, status)
        
    return (all_embeddings_data, status)


# --- Node 1: Build Database on GPU ---
class FaceDB_BuildEmbeddings_GPU:
    MODEL_OPTIONS = ["SFace", "VGG-Face", "Facenet", "Facenet512", "ArcFace"]
    DETECTOR_OPTIONS = ['retinaface', 'mtcnn', 'opencv', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yunet', 'centerface']
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                "directory_path": ("STRING", {"default": "/data/app/input/target_faces"}),
                "db_save_path": ("STRING", {"default": "/data/app/output/face_embeddings_db.pkl"}),
                "model_name": (cls.MODEL_OPTIONS, {"default": "Facenet512"}),
                "detector_backend": (cls.DETECTOR_OPTIONS, {"default": "yolov8"}),
                "force_rebuild": ("BOOLEAN", {"default": True}),
            } }
    RETURN_TYPES = ("FACE_DB", "STRING",)
    RETURN_NAMES = ("face_database", "status_text",)
    FUNCTION = "build_db"
    CATEGORY = "FaceID"
    def build_db(self, **kwargs):
        return _build_db_logic(**kwargs)


# --- Node 2: Find Matching Faces (MODIFIED) ---
class FaceDB_FindMatches:
    DETECTOR_OPTIONS = ['retinaface', 'mtcnn', 'opencv', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yunet', 'centerface']
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_database": ("FACE_DB",),
                # MODIFIED: Changed source_image input to a file path string
                "source_image_path": ("STRING", {"default": "/data/app/input/1.jpg"}),
                "detector_backend": (cls.DETECTOR_OPTIONS, {"default": "yolov8"}),
                "similarity_threshold": ("FLOAT", {"default": 49.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "top_n": ("INT", {"default": 10, "min": 1, "max": 1000}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("best_match_image", "match_image_with_box", "top_n_results_text", "all_filtered_results_text",)
    FUNCTION = "find_matches"
    CATEGORY = "FaceID"

    # MODIFIED: Method signature changed from `source_image` to `source_image_path`
    def find_matches(self, face_database, source_image_path, detector_backend, similarity_threshold, top_n):
        
        def create_blank_image():
            return torch.zeros(1, 64, 64, 3, dtype=torch.float32)

        if not face_database:
            no_match_text = "Error: Face database is empty or invalid."
            return (create_blank_image(), create_blank_image(), no_match_text, no_match_text)

        # MODIFIED: Check if the source image file exists
        if not os.path.exists(source_image_path):
            error_text = f"Error: Source image not found at path: {source_image_path}"
            return (create_blank_image(), create_blank_image(), error_text, error_text)

        model_name = face_database[0].get('model_name', 'SFace')
        
        try:
            # MODIFIED: Directly use the image path with DeepFace.represent
            source_representations = DeepFace.represent(
                img_path=source_image_path, model_name=model_name,
                detector_backend=detector_backend, enforce_detection=True
            )
            if not source_representations:
                 no_match_text = "Error: No face detected in source image."
                 return (create_blank_image(), create_blank_image(), no_match_text, no_match_text)
            source_embedding = source_representations[0]['embedding']
        except Exception as e:
            error_text = f"Error processing source image: {e}"
            return (create_blank_image(), create_blank_image(), error_text, error_text)

        matches = []
        for target_data in face_database:
            distance = dst.find_cosine_distance(np.array(source_embedding), np.array(target_data['embedding']))
            similarity = (1 - distance) * 100
            if similarity >= similarity_threshold:
                match_info = target_data.copy()
                match_info['similarity_percentage'] = similarity
                match_info['distance'] = distance
                matches.append(match_info)

        if not matches:
            no_match_text = f"No matches found above {similarity_threshold:.1f}% threshold. Try lowering the threshold."
            return (create_blank_image(), create_blank_image(), no_match_text, no_match_text)

        sorted_matches_df = pd.DataFrame(matches).sort_values(by='similarity_percentage', ascending=False).reset_index(drop=True)
        
        top_n_df = sorted_matches_df.head(top_n)
        top_n_results_text = f"--- Top {len(top_n_df)} Matches ---\n" + top_n_df[['target_image', 'similarity_percentage', 'distance']].to_string()
        all_filtered_results_text = f"--- All {len(sorted_matches_df)} Matches > {similarity_threshold:.1f}% ---\n" + sorted_matches_df[['target_image', 'similarity_percentage', 'distance']].to_string()
        
        print("\n\n--- FACEID SEARCH RESULTS ---\n", all_filtered_results_text, "\n---------------------------\n")
        
        best_match = sorted_matches_df.iloc[0].to_dict()
        img_pil = Image.open(best_match['image_path']).convert("RGB")
        img_with_box = img_pil.copy()
        draw = ImageDraw.Draw(img_with_box)
        
        facial_area = best_match['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        draw.rectangle([x, y, x+w, y+h], outline="lime", width=5)
        
        return (pil2tensor(img_pil), pil2tensor(img_with_box), top_n_results_text, all_filtered_results_text)


# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "FaceDB_BuildEmbeddings_GPU": FaceDB_BuildEmbeddings_GPU,
    "FaceDB_FindMatches": FaceDB_FindMatches,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDB_BuildEmbeddings_GPU": "Build Face DB (GPU)",
    "FaceDB_FindMatches": "Find Face in Database",
}
