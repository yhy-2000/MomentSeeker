import sys
sys.path.append('./UniIR-main/video_evaluate')

from clip_sf import CLIPScoreFusion
import torch
import torch.nn.functional as F


from PIL import Image
# 先写一个demo推理的脚本
if __name__=='__main__':
    model = CLIPScoreFusion(
            model_name='ViT-L/14',
            download_root='/share/huaying/pretrained_model/vit',
        )
    model.float()
    state_dict = torch.load('/share/huaying/pretrained_model/UniIR/checkpoint/CLIP_SF/clip_sf_large.pth')
    # model.load_state_dict(state_dict['model'], strict=False)
    model.load_state_dict(state_dict['model'])
    model = model.cuda()
    model.eval()

    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()


    text = ["Replace the car with a child", "a photo of a car"]
    text_input = tokenizer(text).long().cuda()
    image = [Image.open('/share/huaying/long_video/car.png'),Image.open('/share/huaying/long_video/girl.png')]
    image_input = [img_preprocess_fn(image).unsqueeze(0) for image in image]
    image_input = torch.cat(image_input, dim=0).cuda()

    text_embed = model.encode_text(text_input).cuda()
    image_embed = model.encode_image(image_input).cuda()
    similarity = text_embed @ image_embed.T
    probability = F.softmax(similarity, dim=0)
    print(probability)

    multimodal_text = ["Replace the car with a child", "a photo of a car", "a photo of a child"]
    multimodal_input_image = [Image.open('/share/huaying/long_video/car.png'),Image.open('/share/huaying/long_video/car.png'),Image.open('/share/huaying/long_video/car.png')]
    output_image = [Image.open('/share/huaying/long_video/car.png'),Image.open('/share/huaying/long_video/girl.png')]
    multimodal_text_input = tokenizer(multimodal_text).long().cuda()
    multimodal_image_input = [img_preprocess_fn(image).unsqueeze(0) for image in multimodal_input_image]
    multimodal_image_input = torch.cat(multimodal_image_input, dim=0).cuda()
    output_image_input = [img_preprocess_fn(image).unsqueeze(0) for image in output_image]
    output_image_input = torch.cat(output_image_input, dim=0).cuda()

    multimodal_embed = model.encode_multimodal_input(multimodal_text_input, multimodal_image_input, torch.ones(len(multimodal_text_input)).cuda(), torch.ones(len(multimodal_image_input)).cuda())
    output_embed = model.encode_image(output_image_input.cuda())
    similarity = multimodal_embed @ output_embed.T
    probability = F.softmax(similarity, dim=1)
    print(probability)

    

    