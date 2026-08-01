import os
import cv2
import numpy as np
import onnxruntime as ort

# Standard reference landmarks for 112x112 face image (Umeyama alignment)
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)

def align_face(image_bgr, landmarks):
    """
    Aligns a face image using 5 landmarks to a canonical 112x112 crop.
    
    Args:
        image_bgr (np.ndarray): Original BGR image.
        landmarks (list or np.ndarray): 5 landmark points [[x, y], ...].
        
    Returns:
        np.ndarray: 112x112 aligned BGR face crop.
    """
    landmarks = np.array(landmarks, dtype=np.float32)
    # Estimate similarity transformation matrix (rotation, scaling, translation)
    M, inliers = cv2.estimateAffinePartial2D(landmarks, REFERENCE_FACIAL_POINTS)
    if M is None:
        # Fallback to simple crop if transform estimation fails
        return cv2.resize(image_bgr, (112, 112))
    aligned = cv2.warpAffine(image_bgr, M, (112, 112))
    return aligned

def nms(dets, thresh):
    """
    Pure Python Non-Maximum Suppression (NMS).
    
    Args:
        dets (np.ndarray): Array of detections [[x1, y1, x2, y2, score], ...].
        thresh (float): NMS threshold.
        
    Returns:
        list: Indices of detections to keep.
    """
    if dets.shape[0] == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

class SCRFDDetector:
    def __init__(self, model_path=None, conf_thresh=0.5, nms_thresh=0.4):
        """
        SCRFD Face Detector implementation using ONNX Runtime.
        """
        if model_path is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            model_path = os.path.join(models_dir, "det_10g.onnx")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SCRFD model weights not found at {model_path}. Please run download script first.")
            
        # Select execution providers (prefer GPU if CUDA is available)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # SCRFD standard strides and anchor counts
        self.strides = [8, 16, 32]
        self.anchor_num = 2

    def _generate_stride_anchors(self, grid_h, grid_w, stride):
        """
        Generates coordinates of anchor centers for a given feature grid.
        """
        y_coords, x_coords = np.mgrid[0:grid_h, 0:grid_w]
        centers = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2) * stride
        # Repeat anchors based on the anchor_num (usually 2 in SCRFD)
        centers = np.repeat(centers, self.anchor_num, axis=0)
        return centers.astype(np.float32)

    def detect(self, img_bgr):
        """
        Detects faces in a BGR image.
        
        Returns:
            list: List of dicts, each containing:
                - 'bbox': [x1, y1, x2, y2, score]
                - 'kps': [[x_eye_l, y_eye_l], [x_eye_r, y_eye_r], [x_nose, y_nose], [x_mouth_l, y_mouth_l], [x_mouth_r, y_mouth_r]]
                - 'score': float
        """
        h_orig, w_orig = img_bgr.shape[:2]
        
        # Preprocessing: resize to 640x640 with letterboxing (padding to preserve aspect ratio)
        input_size = 640
        scale = min(input_size / h_orig, input_size / w_orig)
        h_new, w_new = int(h_orig * scale), int(w_orig * scale)
        
        resized = cv2.resize(img_bgr, (w_new, h_new))
        
        # Create padded 640x640 canvas
        canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        # Pad at top-left
        canvas[:h_new, :w_new, :] = resized
        
        # Convert BGR to RGB (required by InsightFace models)
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Normalization: (img - 127.5) / 128.0
        blob = (img_rgb.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1)) # HWC to CHW
        blob = np.expand_dims(blob, axis=0) # CHW to NCHW
        
        # Run ONNX session
        outputs = self.session.run(None, {self.input_name: blob})
        
        # Group the 9 output tensors by sequence length (which corresponds to feature map sizes of different strides)
        # Strides:
        # 8: 80x80 -> 80*80*2 = 12800 anchors
        # 16: 40x40 -> 40*40*2 = 3200 anchors
        # 32: 20x20 -> 20*20*2 = 800 anchors
        grouped_outputs = {}
        for out in outputs:
            seq_len = out.shape[0]
            if seq_len not in grouped_outputs:
                grouped_outputs[seq_len] = []
            grouped_outputs[seq_len].append(out)
            
        all_bboxes = []
        all_kps = []
        
        # Stride definitions mapped by expected anchor count for 640x640 input
        stride_map = {
            12800: (8, 80, 80),
            3200: (16, 40, 40),
            800: (32, 20, 20)
        }
        
        for seq_len, out_list in grouped_outputs.items():
            if seq_len not in stride_map:
                continue
                
            stride, grid_h, grid_w = stride_map[seq_len]
            anchor_centers = self._generate_stride_anchors(grid_h, grid_w, stride)
            
            # Identify tensors based on second dimension shape
            # shape[1] == 1 -> scores
            # shape[1] == 4 -> bbox offsets
            # shape[1] == 10 -> kps offsets
            scores_raw = None
            bbox_raw = None
            kps_raw = None
            
            for tensor in out_list:
                dim_2 = tensor.shape[1]
                if dim_2 == 1:
                    scores_raw = tensor
                elif dim_2 == 4:
                    bbox_raw = tensor
                elif dim_2 == 10:
                    kps_raw = tensor
                    
            if scores_raw is None or bbox_raw is None or kps_raw is None:
                continue
                
            # The output scores from this ONNX model are already sigmoid-activated probabilities
            scores = scores_raw[:, 0]
            
            # Filter by confidence threshold
            keep_idx = np.where(scores >= self.conf_thresh)[0]
            if len(keep_idx) == 0:
                continue
                
            # Decode bounding boxes
            # bbox output is [l, t, r, b]
            centers = anchor_centers[keep_idx]
            offsets_bbox = bbox_raw[keep_idx] * stride
            
            x1 = centers[:, 0] - offsets_bbox[:, 0]
            y1 = centers[:, 1] - offsets_bbox[:, 1]
            x2 = centers[:, 0] + offsets_bbox[:, 2]
            y2 = centers[:, 1] + offsets_bbox[:, 3]
            
            bboxes = np.stack([x1, y1, x2, y2, scores[keep_idx]], axis=-1)
            all_bboxes.append(bboxes)
            
            # Decode 5 keypoints (Landmarks)
            offsets_kps = kps_raw[keep_idx] * stride
            kps = np.zeros((len(keep_idx), 5, 2), dtype=np.float32)
            for i in range(5):
                kps[:, i, 0] = centers[:, 0] + offsets_kps[:, i*2]
                kps[:, i, 1] = centers[:, 1] + offsets_kps[:, i*2+1]
                
            all_kps.append(kps)
            
        if len(all_bboxes) == 0:
            return []
            
        all_bboxes = np.concatenate(all_bboxes, axis=0)
        all_kps = np.concatenate(all_kps, axis=0)
        
        # Run Non-Maximum Suppression (NMS)
        keep = nms(all_bboxes, self.nms_thresh)
        if len(keep) == 0:
            return []
            
        all_bboxes = all_bboxes[keep]
        all_kps = all_kps[keep]
        
        # Map coordinates back to original image space (remove padding and reverse scaling)
        final_faces = []
        for i in range(len(all_bboxes)):
            bbox = all_bboxes[i]
            kps = all_kps[i]
            
            # Bounding box mapping
            x1 = max(0, int(bbox[0] / scale))
            y1 = max(0, int(bbox[1] / scale))
            x2 = min(w_orig, int(bbox[2] / scale))
            y2 = min(h_orig, int(bbox[3] / scale))
            score = float(bbox[4])
            
            # Landmarks mapping
            mapped_kps = []
            for kp in kps:
                kx = min(w_orig, max(0, int(kp[0] / scale)))
                ky = min(h_orig, max(0, int(kp[1] / scale)))
                mapped_kps.append([kx, ky])
                
            final_faces.append({
                'bbox': [x1, y1, x2, y2, score],
                'kps': mapped_kps,
                'score': score
            })
            
        return final_faces

class ArcFaceEmbedder:
    def __init__(self, model_path=None):
        """
        ArcFace Face Embedding Generator implementation using ONNX Runtime.
        """
        if model_path is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            model_path = os.path.join(models_dir, "w600k_r50.onnx")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace model weights not found at {model_path}. Please run download script first.")
            
        # Select execution providers (prefer GPU if CUDA is available)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, aligned_face_bgr):
        """
        Generates a 512-dimensional normalized embedding for a 112x112 BGR face image.
        
        Args:
            aligned_face_bgr (np.ndarray): 112x112 BGR aligned face chip.
            
        Returns:
            np.ndarray: 512-dimensional float32 vector normalized to unit length.
        """
        assert aligned_face_bgr.shape == (112, 112, 3), "Input image shape must be (112, 112, 3)"
        
        # Convert BGR to RGB (required by ArcFace ONNX model)
        face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        
        # Normalization: (img - 127.5) / 127.5 (standard std and mean for w600k_r50)
        blob = (face_rgb.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1)) # HWC to CHW
        blob = np.expand_dims(blob, axis=0) # CHW to NCHW
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: blob})
        embedding = outputs[0][0] # extract vector
        
        # L2 Normalization (make vector length = 1.0)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
