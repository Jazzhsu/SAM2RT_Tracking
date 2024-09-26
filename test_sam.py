from sam2.build_sam import build_sam2_camera_predictor
import torch
import cv2
import numpy as np

is_init = [False]

class MouseEventListener:
    def __init__(self, model, is_init):
        self._model = model
        self._frame = None
        self._is_init = is_init

    def update_frame(self, frame):
        self._frame = frame

    def on_mouse_clicked(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self._frame is None:
            return

        if not self._is_init[0]:
            self._model.load_first_frame(self._frame)

        self._is_init[0] = True
        # bbox = np.array([[x, y], [x+100, y+100]], dtype=np.float32)
        predictor.add_new_prompt(frame_idx=0, obj_id=1, points=[[x,y]], labels=[1])

ckpt = 'sam2_hiera_tiny.pt'
cfg = 'sam2_hiera_t.yaml'
predictor = build_sam2_camera_predictor(cfg, ckpt)

listener = MouseEventListener(predictor, is_init)

cam = cv2.VideoCapture(0)
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', listener.on_mouse_clicked)


with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        listener.update_frame(frame)
        if not ret:
            continue
    
        width, height = frame.shape[:2][::-1]
        if is_init[0]:
            out_obj_ids, out_mask_logits = predictor.track(frame)
            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            # print(all_mask.shape)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1,2,0).cpu().numpy().astype(np.uint8) * 255
                all_mask = cv2.bitwise_or(all_mask, out_mask)
    
            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            all_mask[:,:,1:] = 0
            frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Camera', frame)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

cam.release()
