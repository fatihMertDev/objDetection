import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn') # Thats for macOS
    class PennFudanDataset(torch.utils.data.Dataset):  
        def __init__(self, root, transforms):
            self.root = root # I need this for variables derived from that class to have the root
            self.transforms = transforms # Same as upper sentence
            
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages")))) # Sorting so my images and masked verison will match
            self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

        def __getitem__(self, idx):
            
            img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
            mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
            img = read_image(img_path)
            mask = read_image(mask_path)
            obj_ids = torch.unique(mask)
            obj_ids = obj_ids[1:] # to remove the background
            num_objs = len(obj_ids)

            masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

            
            boxes = masks_to_boxes(masks) # bounding boxes

            
            labels = torch.ones((num_objs,), dtype=torch.int64) # I am only labeling one object so its ones tensor

            image_id = idx
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            
            img = tv_tensors.Image(img) # wrapped

            target = {}
            target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
            target["masks"] = tv_tensors.Mask(masks)
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd  # that whole target part for my aim attributes for machine to learn

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        def __len__(self):
            return len(self.imgs) 
        

    backbone = torchvision.models.resnet50(weights= "IMAGENET1K_V1")
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
    backbone.out_channels = 2048

    generator_sizes = ((32, 64, 128, 256, 512),)
    aspect_ratios = ((0.5, 1.0, 2.0),)
    anchor_generator = AnchorGenerator(sizes=generator_sizes, aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names = ["0"],
                                                    output_size = 7,
                                                    sampling_ratio = 2,)

    model = FasterRCNN(backbone,
                    num_classes=2,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,
                    )

    device = torch.device("cpu")
    model.to(device)


    dataset_root = os.path.join(os.path.dirname(__file__), "PennFudanPed") # had some problems with the directory so i made it this way so it will find wherever it is

    dataset = PennFudanDataset(root=dataset_root, transforms=None) # could make some transforms but maybe later i will try
    def collate_fn(batch):
        return tuple(zip(*batch)) # making tuple of minibatches 

    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )


    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=0.005,  
        momentum=0.9,  
        weight_decay=0.0005, 
    )

    model.eval()  # evaluation mode activated so autograd will be turned off 

    # that is the part where boxes are being drawn according to perdictions
    for images, targets in data_loader:
        images = [img.to(device).to(torch.float32) / 255.0 for img in images]

        try:
            with torch.no_grad():
                predictions = model(images)
            
            print(predictions)

            if len(predictions) > 0 and 'boxes' in predictions[0]:
                pred_boxes = predictions[0]['boxes'].cpu().numpy()
                img = images[0].cpu()
                
                # Görselleştirme
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(TF.to_pil_image(img))

                for box in pred_boxes:
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)

                plt.show(block=False)
            else:
                print("No predictions found or no bounding boxes.")
        
        except Exception as e:
            print(f"Error occurred: {e}")

        break  
        
    

    

    
    


   

        