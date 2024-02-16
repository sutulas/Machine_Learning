# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:58:09 2023

@author: sutul
"""

# From your collection of personal photographs, pick 10 images of an￾imals 
# (such as dogs, cats, birds, farm animals, etc.). If the subject
# does not occupy a reasonable part of the image, then crop the image.
# Now use an image classifcation CNN to
# predict the class of each of your images, and report the probabilities
# for the top fve predicted classes for each image

from dl_imports import *

resize = Resize((232, 232), antialias=True)
crop = CenterCrop(224)
normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#imgfiles = sorted([f for f in glob("images/*")])
imgfiles = ['images/1.jpg', 'images/2.jpg', 'images/3.JPG', 
            'images/4.JPG', 'images/5.jpg']
imgs = torch.stack(
    [torch.div(crop(resize(read_image(f))), 255) for f in imgfiles]
)
imgs = normalize(imgs)
imgs.size()

resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
print(
    summary(
        resnet_model,
        input_data=imgs,
        col_names=["input_size", "output_size", "num_params"],
    )
)

resnet_model.eval()

img_preds = resnet_model(imgs)

img_probs = np.exp(np.asarray(img_preds.detach()))
img_probs /= img_probs.sum(1)[:, None]

labs = json.load(open("imagenet_class_index.json"))
class_labels = pd.DataFrame(
    [(int(k), v[1]) for k, v in labs.items()], columns=["idx", "label"]
)
class_labels = class_labels.set_index("idx")
class_labels = class_labels.sort_index()

for i, imgfile in enumerate(imgfiles):
    img_df = class_labels.copy()
    img_df["prob"] = img_probs[i]
    img_df = img_df.sort_values(by="prob", ascending=False)[:5]
    print(f"\nImage: {imgfile}")
    print(img_df.reset_index().drop(columns=["idx"]))


def classify_image(image_file, n=10):
    imgs = torch.stack([torch.div(crop(resize(read_image(image_file))), 255)])
    img_preds = resnet_model(imgs)
    img_probs = np.exp(np.asarray(img_preds.detach()))
    img_probs /= img_probs.sum(1)[:, None]
    img_df = class_labels.copy()
    img_df["prob"] = img_probs[0]
    img_df = img_df.sort_values(by="prob", ascending=False)[:n]
    print(f"\nImage: {imgfile}")
    print(img_df.reset_index().drop(columns=["idx"]))
    
    
#################
#   results     #
#################

#image 1 is a go-kart
#image 2 is a water pistol
#image 3 is a gecko-thing
#image 4 is a toy duck
#image 5 is a dog


#the model predicted the following:

#image1: go-kart 0.1076
#image2: screwdriver 0.1234
#image3: banded_gecko 0.2994
#image4: plastic_bag 0.0960
#image5: American_Staffordshire_terrier 0.256948

#i am pretty impressed with this model, it relatively correctly classified 3/5
#images, and the 2 it got incorrect were fairly niche categories it probably
#did not come across very often in training. 
#I could probably get better results by using more standard pictures, but I
#am impressed nonetheless
