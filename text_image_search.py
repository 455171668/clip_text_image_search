import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO
import clip
import json
import os

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# 定义图像预处理函数
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])

# 设置图片报错信息
with open('/var/lib/docker/lzd/ovd-lp/data/coco/annotations/instances_val2017.json') as f:
    data = json.load(f)
imgs = data['images']
img_ids = [img['id'] for img in imgs]
image_paths = []

# 定义搜索函数
def search_images(query, k=5):
    print(f"搜索图片开始：搜索关键词为 {query}")
    # 对查询进行编码
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(query).to(device)).float()

    # 计算每个图像和查询之间的相似度
    print(f"搜索图片开始：计算文字和图片相似度进度")
    similarities = []
    for i,(img_id) in enumerate(img_ids):
        path = f'/var/lib/docker/lzd/ovd-lp/data/coco/val2017/{img_id:012}.jpg'
        image_paths.append(path)
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                image_encoded = model.encode_image(image.to(device)).float()
                similarity = (100.0 * text_encoded @ image_encoded.T).item()
                similarities.append(similarity)
            if (i+1)%500 ==0 :
                print(f"搜索图片进行中：计算文字和图片相似度进度：{i+1}/{len(img_ids)}")
    # 获取前k个最相似的图像的索引
    idxs = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]

    # 打印结果
    print(f"搜索图片结束：开始打印搜索结果")
    output_folder = 'SearchResult'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, idx in enumerate(idxs):
        # 加载图像并打印文件名和相似度
        image_path = image_paths[idx]
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img.save(os.path.join(output_folder, f"{i}.jpg"))
        print(f"搜索图片结果：{i+1}. {image_path} ({similarities[idx]:.2f})")
    print(f"搜索图片结束：图片保存到SearchResult文件夹")

# 搜索图像
search_images("cat and dog")