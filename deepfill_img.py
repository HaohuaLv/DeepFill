import argparse
import network
import torch
import cv2
import numpy as np

drawing = False
brush_size = 100
prev_x, prev_y = None, None
mask_brush = 255

def deepfill(img_org, generator):

    global prev_x, prev_y, drawing, brush_size, mask_brush
    def render(img_org, mask):

        preview_img = img_org.copy()
        if mask_brush == 255:
            preview_img[np.where(mask == 255)] = 255
            cv2.circle(preview_img, (prev_x, prev_y), brush_size, (255, 255, 255), -1)
        elif mask_brush == 0:
            mask_copy = mask.copy()
            cv2.circle(mask_copy, (prev_x, prev_y), brush_size, (0,), -1)
            preview_img[np.where(mask_copy == 255)] = 255
            cv2.circle(preview_img, (prev_x, prev_y), brush_size, (255, 255, 255), 2)
        return preview_img

    def draw_mask(event, x, y, flags, param):
        global prev_x, prev_y, drawing, brush_size, mask_brush

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            prev_x, prev_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.circle(mask, (x, y), brush_size, (mask_brush,), -1)
                cv2.line(mask, (prev_x, prev_y), (x, y), (mask_brush,), brush_size)
            prev_x, prev_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(mask, (x, y), brush_size, (mask_brush,), -1)

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                brush_size += 2
            else:
                brush_size -= 2
            if brush_size < 2:
                brush_size = 2

    mask = np.zeros(img_org.shape[:2], dtype=np.uint8)
    img_deepfill = img_org.copy()

    cv2.namedWindow('Image_masked')
    cv2.setMouseCallback('Image_masked', draw_mask)
    cv2.imshow('Preview', img_deepfill)
    
    while True:
        preview_img = render(img_org, mask)
        cv2.imshow('Image_masked', preview_img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:     # Press ESC
            break

        elif k == 13:   # Press Enter
            img_masked = img_org.copy()
            img_masked[np.where(mask == 255)] = 255
            cv2.imwrite('./output.png', img_deepfill)
            cv2.imwrite('./img_masked.png', img_masked)
            break

        elif k == 9:    # Press Table
            img = img_org.copy()
            mask_ = mask.copy()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
            mask_ = torch.from_numpy(mask_.astype(np.float32) / 255.0).unsqueeze(0).contiguous()

            img = img.unsqueeze(0).cuda()
            mask_ = mask_.unsqueeze(0).cuda()

            with torch.no_grad():
                first_out, second_out = generator(img, mask_)

            first_out_wholeimg = img * (1 - mask_) + first_out * mask_
            second_out_wholeimg = img * (1 - mask_) + second_out * mask_

            second_out_wholeimg = second_out_wholeimg * 255
            img_deepfill = second_out_wholeimg.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
            img_deepfill = np.clip(img_deepfill, 0, 255)
            img_deepfill = img_deepfill.astype(np.uint8)
            img_deepfill = cv2.cvtColor(img_deepfill, cv2.COLOR_RGB2BGR)
            cv2.imshow('Preview', img_deepfill)

        elif (k == 49) | (k == 50):
            mask_brush = 255 if k == 49 else 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type = int, default = 4, help = 'input RGB image + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
    parser.add_argument('--latent_channels', type = int, default = 48, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'elu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    opt = parser.parse_args()
    
    
    generator = network.GatedGenerator(opt).eval().cuda()
    generator.load_state_dict(torch.load('./deepfillv2.pth'))

    img_org = cv2.imread('./input.png')
    deepfill(img_org, generator)
