import numpy as np
import torch
from PIL import Image

# FrameMaker 0.1.0 by diSty

class FrameMaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'subject_image': ('IMAGE',),
                'frame_count': ('INT', {'default': 4}),
                "invert_alpha": (['true','false'], {"default": 'false'}),
                "movement_presets": (movement,{"default": 'left to center',}),
                'movement_presets_distance': ('INT', {'default': 50, 'min': -10000, 'max': 10000}),
                "movement_direction": (['None','front', 'back'],{"default": 'back',}),
                'x_pos_start': ('INT', {'default': 0, 'min': -10000, 'max': 10000}),
                'x_pos_end': ('INT', {'default': 0, 'min': -10000, 'max': 10000}),
                'y_pos_start': ('INT', {'default': 0, 'min': -10000, 'max': 10000}),
                'y_pos_end': ('INT', {'default': 0, 'min': -10000, 'max': 10000}),
                'scale_start': ('FLOAT', {'default': 1, 'min': 0.05, 'step': 0.05, 'max': 10}),
                'scale_end': ('FLOAT', {'default': 1, 'min': 0.05, 'step': 0.05, 'max': 10}),
                "scale_anchor_point": (['center','top center', 'bottom center', 'top left', 'top right', 'bottom left', 'bottom right'],),
            },
            "optional": {
                "subject_alpha": ("MASK",),
                'bg1': ('IMAGE',),
                'bg2': ('IMAGE',),
                'bg3': ('IMAGE',),
                'bg4': ('IMAGE',),
                'frames': ('IMAGE',),    
                "resize_frame_input": (['true','false'], {"default": 'true'}),
                'frame_input_width': ('INT', {'default': 768, 'min': 16,'step': 16,  'max': 8192}),
                'frame_input_height': ('INT', {'default': 768, 'min': 16,'step': 16,  'max': 8192}),
                "resize_frame_output": (['true','false'], {"default": 'false'}),
                'frame_output_width': ('INT', {'default': 512, 'min': 16,'step': 16,  'max': 8192}),
                'frame_output_height': ('INT', {'default': 512, 'min': 16,'step': 16,  'max': 8192}),
                "invert_order": (['true','false'], {"default": 'false'}),
                # "limit_frames": (['true','false'], {"default": 'false'}),
                # 'limit_frames_count': ('INT', {'default': 4, 'min': 1,'step': 1,  'max': 10000}),
                # 'limit_frames_index': ('INT', {'default': 1, 'min': 1,'step': 1,  'max': 10000}),
            }
        }     

    RETURN_TYPES = ('IMAGE','IMAGE','INT', 'INT', 'INT',)
    RETURN_NAMES = ('frames','subject_image', 'width', 'height','frame_count',)
    FUNCTION = 'process_frames'
    CATEGORY = '🅓 diSty/FrameMaker'

    def process_frames(self, subject_image, frame_count,invert_alpha,movement_presets,movement_presets_distance,movement_direction,
                        x_pos_start, x_pos_end, y_pos_start, y_pos_end, scale_start, scale_end, scale_anchor_point,
                        subject_alpha=None,bg1=None, bg2=None, bg3=None, bg4=None,frames=None,width=0, height=0,resize_frame_input=0, frame_input_width=0,
                        frame_input_height=0,resize_frame_output=None, frame_output_width=0,frame_output_height=0,invert_order='false'
                        # , **kwargs
                        ):

        if subject_alpha is not None:
            if invert_alpha == 'true':
                subject_alpha = 1.0 - subject_alpha
            subject_image = join_image_with_alpha(subject_image, subject_alpha)

        foreground_image_pil = tensor2pil(subject_image)
        backgrounds = [bg1, bg2, bg3, bg4]
        backgrounds = [b for b in backgrounds if b is not None]

        if frames is not None:
            backgrounds = []
            for i in range(frames.shape[0]):
                frame_tensor = frames[i]
                backgrounds.append(frame_tensor) 

        if resize_frame_input == 'true':
            foreground_image_pil = foreground_image_pil.resize((frame_input_width, frame_input_height), Image.Resampling.LANCZOS)
            width = frame_input_width
            height = frame_input_height

            resized_backgrounds_pil = []
            for bg_tensor in backgrounds:
                bg_pil = tensor2pil(bg_tensor)
                resized_bg_pil = bg_pil.resize((frame_input_width, frame_input_height), Image.Resampling.LANCZOS)
                resized_backgrounds_pil.append(resized_bg_pil)            

        movement_map = movement_map_f(movement_presets_distance)

        if movement_presets != 'None' and movement_presets in movement_map:
            preset = movement_map[movement_presets]
            x_pos_start = preset['x_start']
            x_pos_end = preset['x_end']
            y_pos_start = preset['y_start']
            y_pos_end = preset['y_end']        

        x_pos_start_pix = x_pos_start * foreground_image_pil.width / 100
        x_pos_end_pix = x_pos_end * foreground_image_pil.width / 100
        y_pos_start_pix = y_pos_start * foreground_image_pil.height / 100
        y_pos_end_pix = y_pos_end * foreground_image_pil.height / 100

        if movement_direction != 'None':
            if movement_direction == 'front':
                scale_start = .3
                scale_end  = 1.3
            if movement_direction == 'back':
                scale_start = 1.3
                scale_end  = .3

        x_step = (x_pos_end_pix - x_pos_start_pix) / (frame_count - 1)
        y_step = (y_pos_end_pix - y_pos_start_pix) / (frame_count - 1)
        scale_step = (scale_end - scale_start) / (frame_count - 1)

        result_images = []
        for frame in range(frame_count):
            if resize_frame_input == 'true':
                background_image_pil = resized_backgrounds_pil[frame % len(resized_backgrounds_pil)]
            else:
                background_image_pil = tensor2pil(backgrounds[frame % len(backgrounds)])

            scale = scale_start + frame * scale_step
            scaled_width = int(foreground_image_pil.width * scale)
            scaled_height = int(foreground_image_pil.height * scale)

            foreground_resized = foreground_image_pil.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

            x_pos = x_pos_start_pix + frame * x_step
            y_pos = y_pos_start_pix + frame * y_step

            width_diff = (scaled_width - foreground_image_pil.width)
            height_diff = (scaled_height - foreground_image_pil.height)

            if 'center' in scale_anchor_point.lower():
                x_pos -= width_diff // 2
            if scale_anchor_point.lower() == 'center':
                y_pos -= height_diff // 2
            elif 'bottom' in scale_anchor_point.lower():
                y_pos -= height_diff
            if 'right' in scale_anchor_point.lower():
                x_pos -= width_diff
            frame_image = background_image_pil.copy()
            if foreground_resized.mode != 'RGBA':
                foreground_resized = foreground_resized.convert('RGBA')

            frame_image.paste(foreground_resized, (int(x_pos), int(y_pos)), foreground_resized)

            if resize_frame_output == 'true':
                frame_image = frame_image.resize((frame_output_width, frame_output_height), Image.Resampling.LANCZOS)
                width = frame_output_width
                height = frame_output_height
            result_images.append(pil2tensor(frame_image))

        batch = torch.cat(result_images, dim=0)
        if invert_order == 'true':
            batch = batch.flip(dims=[0])      

        return (batch,) + (subject_image,) + (width,) + (height,) + (frame_count,)
    
class FrameMakerBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'frames': ('IMAGE',),    
                'subject_image': ('IMAGE',),
                'frame_count': ('INT', {'default': 4}),
                "invert_alpha": (['true','false'], {"default": 'false'}),
                "movement_presets": (movement,{"default": 'left to center',}),
                'movement_presets_distance': ('INT', {'default': 50, 'min': -10000, 'max': 10000}),
                "movement_direction": (['None','front', 'back'],{"default": 'back',}),
                'x_pos_start': ('INT', {'default': 0, 'min': -10000, 'max': 10000}),
                'x_pos_end': ('INT', {'default': 0, 'min': -10000, 'max': 10000}),
                'y_pos_start': ('INT', {'default': 0, 'min': -10000, 'max': 10000}),
                'y_pos_end': ('INT', {'default': 0, 'min': -10000, 'max': 10000}),
                'scale_start': ('FLOAT', {'default': 1, 'min': 0.05, 'step': 0.05, 'max': 10}),
                'scale_end': ('FLOAT', {'default': 1, 'min': 0.05, 'step': 0.05, 'max': 10}),
                "scale_anchor_point": (['center','top center', 'bottom center', 'top left', 'top right', 'bottom left', 'bottom right'],),
            },
            "optional": {
                "subject_alpha": ("MASK",),
                "resize_frame_input": (['true','false'], {"default": 'true'}),
                'frame_input_width': ('INT', {'default': 768, 'min': 16,'step': 16,  'max': 8192}),
                'frame_input_height': ('INT', {'default': 768, 'min': 16,'step': 16,  'max': 8192}),
                "resize_frame_output": (['true','false'], {"default": 'false'}),
                'frame_output_width': ('INT', {'default': 512, 'min': 16,'step': 16,  'max': 8192}),
                'frame_output_height': ('INT', {'default': 512, 'min': 16,'step': 16,  'max': 8192}),
                "invert_order": (['true','false'], {"default": 'false'}),
            }
        }     

    RETURN_TYPES = ('IMAGE','IMAGE','INT', 'INT', 'INT',)
    RETURN_NAMES = ('frames','subject_image', 'width', 'height','frame_count',)
    FUNCTION = 'process_frames'
    CATEGORY = '🅓 diSty/FrameMaker'

    def process_frames(self,frames,subject_image, frame_count,invert_alpha,movement_presets,movement_presets_distance,movement_direction,
                        x_pos_start, x_pos_end, y_pos_start, y_pos_end, scale_start, scale_end, scale_anchor_point,
                        subject_alpha=None,width=0, height=0,resize_frame_input=0, frame_input_width=0,
                        frame_input_height=0,resize_frame_output=None, frame_output_width=0,frame_output_height=0,invert_order='false'):
        

        if subject_alpha is not None:
            if invert_alpha == 'true':
                subject_alpha = 1.0 - subject_alpha            
            subject_image = join_image_with_alpha(subject_image, subject_alpha)

        foreground_image_pil = tensor2pil(subject_image)

        if frames is not None:
            backgrounds = []
            for i in range(frames.shape[0]):
                frame_tensor = frames[i]
                backgrounds.append(frame_tensor) 

        if resize_frame_input == 'true':
            foreground_image_pil = foreground_image_pil.resize((frame_input_width, frame_input_height), Image.Resampling.LANCZOS)
            width = frame_input_width
            height = frame_input_height

            resized_backgrounds_pil = []
            for bg_tensor in backgrounds:
                bg_pil = tensor2pil(bg_tensor)
                resized_bg_pil = bg_pil.resize((frame_input_width, frame_input_height), Image.Resampling.LANCZOS)
                resized_backgrounds_pil.append(resized_bg_pil)            

        movement_map = movement_map_f(movement_presets_distance)

        if movement_presets != 'None' and movement_presets in movement_map:
            preset = movement_map[movement_presets]
            x_pos_start = preset['x_start']
            x_pos_end = preset['x_end']
            y_pos_start = preset['y_start']
            y_pos_end = preset['y_end']        

        
        x_pos_start_pix = x_pos_start * foreground_image_pil.width / 100
        x_pos_end_pix = x_pos_end * foreground_image_pil.width / 100
        y_pos_start_pix = y_pos_start * foreground_image_pil.height / 100
        y_pos_end_pix = y_pos_end * foreground_image_pil.height / 100

        if movement_direction != 'None':
            if movement_direction == 'front':
                scale_start = .3
                scale_end  = 1.3
            if movement_direction == 'back':
                scale_start = 1.3
                scale_end  = .3

        x_step = (x_pos_end_pix - x_pos_start_pix) / (frame_count - 1)
        y_step = (y_pos_end_pix - y_pos_start_pix) / (frame_count - 1)
        scale_step = (scale_end - scale_start) / (frame_count - 1)

        result_images = []
        for frame in range(frame_count):
            if resize_frame_input == 'true':
                background_image_pil = resized_backgrounds_pil[frame % len(resized_backgrounds_pil)]
            else:
                background_image_pil = tensor2pil(backgrounds[frame % len(backgrounds)])

            scale = scale_start + frame * scale_step
            scaled_width = int(foreground_image_pil.width * scale)
            scaled_height = int(foreground_image_pil.height * scale)

            foreground_resized = foreground_image_pil.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

            x_pos = x_pos_start_pix + frame * x_step
            y_pos = y_pos_start_pix + frame * y_step

            width_diff = (scaled_width - foreground_image_pil.width)
            height_diff = (scaled_height - foreground_image_pil.height)

            if 'center' in scale_anchor_point.lower():
                x_pos -= width_diff // 2

            if scale_anchor_point.lower() == 'center':
                y_pos -= height_diff // 2
            elif 'bottom' in scale_anchor_point.lower():
                y_pos -= height_diff
            if 'right' in scale_anchor_point.lower():
                x_pos -= width_diff
            frame_image = background_image_pil.copy()
            if foreground_resized.mode != 'RGBA':
                foreground_resized = foreground_resized.convert('RGBA')

            frame_image.paste(foreground_resized, (int(x_pos), int(y_pos)), foreground_resized)

            if resize_frame_output == 'true':
                frame_image = frame_image.resize((frame_output_width, frame_output_height), Image.Resampling.LANCZOS)
                width = frame_output_width
                height = frame_output_height
            result_images.append(pil2tensor(frame_image))

        batch = torch.cat(result_images, dim=0)
        if invert_order == 'true':
            batch = batch.flip(dims=[0])      

        return (batch,) + (subject_image,) + (width,) + (height,) + (frame_count,)
            
NODE_CLASS_MAPPINGS = {
    "FrameMaker": FrameMaker,
    "FrameMakerBatch": FrameMakerBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameMaker": f"🅓 Frame Maker ",
    "FrameMakerBatch": f"🅓 Frame Maker Batch"
}

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_mask(mask, shape):
    return torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[0], shape[1]), mode="bilinear").squeeze(1)

def join_image_with_alpha(image: torch.Tensor, alpha: torch.Tensor):
    batch_size = min(len(image), len(alpha))
    out_images = []

    alpha = 1.0 - resize_mask(alpha, image.shape[1:])
    for i in range(batch_size):
        out_images.append(torch.cat((image[i][:,:,:3], alpha[i].unsqueeze(2)), dim=2))

    result = (torch.stack(out_images))
    return result

def movement_map_f(movement_presets_distance):
    movement_map = {
        'left to center': {'x_start': -movement_presets_distance, 'x_end': 0, 'y_start': 0, 'y_end': 0},
        'right to center': {'x_start': movement_presets_distance, 'x_end': 0, 'y_start': 0, 'y_end': 0},
        'center to left': {'x_start': 0, 'x_end': -movement_presets_distance, 'y_start': 0, 'y_end': 0},
        'center to right': {'x_start': 0, 'x_end': movement_presets_distance, 'y_start': 0, 'y_end': 0},
        'left to right': {'x_start': -movement_presets_distance, 'x_end': movement_presets_distance, 'y_start': 0, 'y_end': 0},
        'right to left': {'x_start': movement_presets_distance, 'x_end': -movement_presets_distance, 'y_start': 0, 'y_end': 0},
        'top to center': {'x_start': 0, 'x_end': 0, 'y_start': -movement_presets_distance, 'y_end': 0},
        'bottom to center': {'x_start': 0, 'x_end': 0, 'y_start': movement_presets_distance, 'y_end': 0},
        'center to top': {'x_start': 0, 'x_end': 0, 'y_start': 0, 'y_end': -movement_presets_distance},
        'center to bottom': {'x_start': 0, 'x_end': 0, 'y_start': 0, 'y_end': movement_presets_distance},
        'top to bottom': {'x_start': 0, 'x_end': 0, 'y_start': -movement_presets_distance, 'y_end': movement_presets_distance},
        'bottom to top': {'x_start': 0, 'x_end': 0, 'y_start': movement_presets_distance, 'y_end': -movement_presets_distance},
        'top left to center': {'x_start': -movement_presets_distance, 'x_end': 0, 'y_start': -movement_presets_distance, 'y_end': 0},
        'top right to center': {'x_start': movement_presets_distance, 'x_end': 0, 'y_start': -movement_presets_distance, 'y_end': 0},
        'bottom left to center': {'x_start': -movement_presets_distance, 'x_end': 0, 'y_start': movement_presets_distance, 'y_end': 0},
        'bottom right to center': {'x_start': movement_presets_distance, 'x_end': 0, 'y_start': movement_presets_distance, 'y_end': 0},
        'center to top left': {'x_start': 0, 'x_end': -movement_presets_distance, 'y_start': 0, 'y_end': -movement_presets_distance},
        'center to top right': {'x_start': 0, 'x_end': movement_presets_distance, 'y_start': 0, 'y_end': -movement_presets_distance},
        'center to bottom left': {'x_start': 0, 'x_end': -movement_presets_distance, 'y_start': 0, 'y_end': movement_presets_distance},
        'center to bottom right': {'x_start': 0, 'x_end': movement_presets_distance, 'y_start': 0, 'y_end': movement_presets_distance},
        'top left to bottom right': {'x_start': -movement_presets_distance, 'x_end': movement_presets_distance, 'y_start': -movement_presets_distance, 'y_end': movement_presets_distance},
        'bottom right to top left': {'x_start': movement_presets_distance, 'x_end': -movement_presets_distance, 'y_start': movement_presets_distance, 'y_end': -movement_presets_distance},
        'top right to bottom left': {'x_start': movement_presets_distance, 'x_end': -movement_presets_distance, 'y_start': -movement_presets_distance, 'y_end': movement_presets_distance},
        'bottom left to top right': {'x_start': -movement_presets_distance, 'x_end': movement_presets_distance, 'y_start': movement_presets_distance, 'y_end': -movement_presets_distance},
    }
    
    return movement_map

movement = ['None',
'left to center',
'right to center',
'center to left',
'center to right',
'left to right',
'right to left',

'top to center',
'bottom to center',
'top to bottom',
'bottom to top',
'center to top', 
'center to bottom',

'top left to center', 
'top right to center', 
'bottom left to center', 
'bottom right to center',

'center to top left',
'center to top right',
'center to bottom left', 
'center to bottom right',

'top left to bottom right', 
'bottom right to top left', 
'top right to bottom left',
'bottom left to top right'
]
