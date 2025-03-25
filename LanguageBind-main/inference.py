import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

if __name__ == '__main__':
      device = 'cuda:0'
      device = torch.device(device)
      clip_type = {
            'video': '/share/huaying/pretrained_model/LanguageBind_Video_FT',
            'image': '/share/huaying/pretrained_model/LanguageBind_Image',
      }
      model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
      model = model.to(device)
      model.eval()
      pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
      tokenizer = LanguageBindImageTokenizer.from_pretrained(clip_type['image'])
      modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

      import json

      data_li = json.load(open('./languagebind/benchmark/test/test_k400_cls.json'))
      video = [dic['qry_video_path'] for dic in data_li][:10]
      language = [dic['tgt_text'] for dic in data_li][:10]
      print(language,video)
      inputs = {
            'video': to_device(modality_transform['video'](video), device),
      }

      with torch.no_grad():
            embeddings = model(inputs)

      video_embeddings = embeddings['video']


      inputs = {'language':to_device(tokenizer(language, max_length=77, padding='max_length',
                                                truncation=True, return_tensors='pt'), device)}

      language_embeddings = model(inputs)['language']

      # print(video_embeddings[0].shape)
      # print(video_embeddings[0])

      # print("Video x Text: \n",
      #       torch.softmax(video_embeddings @ language_embeddings.T, dim=-1).detach().cpu().numpy())
      

