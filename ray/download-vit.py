from transformers import AutoImageProcessor, ViTMAEForPreTraining

ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
AutoImageProcessor.from_pretrained('facebook/vit-mae-base')