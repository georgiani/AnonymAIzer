from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import gradio as gr

def show_bb(img, preds, score_lim = 0.3, chosen_blur =None):
    pred = preds[0]
    boxes = pred["boxes"]
    scores = pred["scores"]

    pil = img
    pildraw = ImageDraw.Draw(pil)

    boxes_cnt = 0

    for i, b in enumerate(boxes):
        if scores[i] >= score_lim:
            xmin = round(b[0].item())
            ymin = round(b[1].item())
            xmax = round(b[2].item())
            ymax = round(b[3].item())

            if chosen_blur:
                if boxes_cnt in chosen_blur:
                    mask = Image.new('L', pil.size, 0)
                    draw = ImageDraw.Draw(mask)
                    draw.rectangle([(xmin, ymin), (xmax, ymax) ], fill=255)
                    blurred = pil.filter(ImageFilter.GaussianBlur(30))
                    pil.paste(blurred, mask=mask)
            else:
                pildraw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
                font = ImageFont.truetype("Roboto-Light.ttf", 20)
                pildraw.text((xmin, ymin), str(boxes_cnt), font=font, align ="left", fill="black") 

            boxes_cnt += 1
    return pil, boxes_cnt
    

net = fasterrcnn_resnet50_fpn()
net.roi_heads.box_predictor = FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features, 2)
net.load_state_dict(torch.load("faster_rcnn_10.h5", map_location=torch.device('cpu')))
transform = A.Compose([A.Normalize(mean=0, std=1, always_apply=True), ToTensorV2(p=1.0)])
prediction = None

def predict(img, confidence):
    if not confidence:
        confidence = 0.5
    transformed_image = transform(image=np.array(img.convert("L")))["image"].unsqueeze(0)
    
    net.eval()
    with torch.no_grad():
        global prediction
        prediction = net(transformed_image)
        with_bb, bbs = show_bb(img, prediction, confidence)
        return [
            with_bb, 
            gr.Slider(0, 1, value=0.5, label="Confidence", visible=True),
            gr.CheckboxGroup([i for i in range(bbs)], interactive=True)
        ]

def change_confid(img, confidence):
    with_bb, bbs = show_bb(img, prediction, confidence)
    return [
        with_bb,
        gr.CheckboxGroup([i for i in range(bbs)], interactive=True)
    ]

def blur_image(img, bb_to_blur, confidence):
    with_bb, _ = show_bb(img, prediction, confidence, bb_to_blur)
    return with_bb

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil")
            run_button = gr.Button("Run")
        output_bb = gr.Image(interactive=False)
        
        with gr.Column():
            blurred_image = gr.Image(interactive=False)
            render_button = gr.Button("Render")

    confid = gr.Slider(0, 1, value=0.5, label="Confidence", visible=False)

    # TODO: have checkboxes for every bb, then on hover show the bb
    # on the picture in output, then if checked, blur the bb
    pred_list = gr.CheckboxGroup([], interactive=True)

    run_button.click(
        predict, 
        [input_image, confid], 
        [output_bb, confid, pred_list]
    )

    confid.change(
        change_confid, 
        [input_image, confid], 
        [output_bb, pred_list]
    )

    render_button.click(
        blur_image,
        [input_image, pred_list, confid],
        blurred_image
    )

if __name__ == "__main__":
    demo.queue().launch(show_api=False)