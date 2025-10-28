from .mlrgcn import Mlrgcn, mlrgcn
from .model_builder import build_model

def mlrgcn_wrapper(num_classes, **kwargs):
    """Real MLR-GCN implementation using existing COCO dataset structure"""
    import torch
    import os
    from pathlib import Path
    
    # Create config object with required attributes for COCO
    class Config:
        class MODEL:
            BACKBONE = type('obj', (object,), {'NAME': 'ViT-B/32'})
        class DATASET:
            NAME = 'coco'
        class TRAINER:
            class COOP_MLC:
                N_CTX_POS = 4
                N_CTX_NEG = 4
                POSITIVE_PROMPT_INIT = ""
                NEGATIVE_PROMPT_INIT = ""
                CSC = False
                LS = 0.2
            FINETUNE_BACKBONE = False
            FINETUNE_ATTN = False
        USE_CUDA = True
        path_to_relation = "relation/co_occurrence_matrix_coco.pth"
    
    cfg = Config()
    
    # Get COCO class names from existing dataset structure
    try:
        from gmean_mlc.datasets.dataset_coco import MultiLabelDatasetCoco
        import torchvision.transforms as transforms
        
        # Create a temporary dataset to get class names (same as TresNet/MaxViT use)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        temp_dataset = MultiLabelDatasetCoco(
            annRoot="/mnt/datassd3/coco-2017/output_balanced_70_15_15/",
            imgRoot="/mnt/datassd3/coco-2017/images/all_images/",
            split="Train",
            transform=transform,
        )
        
        classnames = temp_dataset.LabelNames
        print(f"✓ Loaded {len(classnames)} COCO class names from your curated dataset")
        
    except Exception as e:
        print(f"Warning: Could not load COCO class names: {e}")
        # Fallback to default COCO class names
        classnames = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        print(f"✓ Using default COCO class names ({len(classnames)} classes)")
    
    # Ensure co-occurrence matrix exists
    relation_path = Path(cfg.path_to_relation)
    if not relation_path.exists():
        print(f"Creating COCO co-occurrence matrix at {relation_path}")
        relation_path.parent.mkdir(exist_ok=True)
        
        # Create the matrix using existing script
        from .create_relation_matrices import create_coco_relation_matrix
        coco_matrix = create_coco_relation_matrix()
        torch.save(coco_matrix, relation_path)
        print(f"✓ Created COCO relation matrix: {coco_matrix.shape}")
    
    # Use the real MLR-GCN implementation
    print("Building real MLR-GCN with CLIP integration...")
    model = mlrgcn(cfg, classnames)
    
    return model

__all__ = ['Mlrgcn', 'mlrgcn', 'mlrgcn_wrapper', 'build_model']