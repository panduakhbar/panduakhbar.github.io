import os
import geoai
from IPython.display import display


weights_path = "/content/drive/MyDrive/RFDETR/checkpoint_best_total.pth"
image_path = "/content/drive/MyDrive/RFDETR/tes_ortho6.tif"

print("Weights exists:", os.path.exists(weights_path))
print("Image exists:", os.path.exists(image_path))

class_names = ["palm_tree"]

gdf = geoai.rfdetr_detect(
    input_path=image_path,
    model_variant="nano",  # harus sama dengan model saat training
    pretrain_weights=weights_path,
    confidence_threshold=0.3,
    nms_threshold=0.3,
    class_names=class_names,
    device="cpu",
    batch_size=1,
)

print(f"Detected {len(gdf)} objects")
display(gdf.head(10))
