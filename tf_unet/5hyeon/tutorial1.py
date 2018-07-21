# Following the tutorial
# set pascal VOC as a db

# purpose of current tutorial
# 1. get use to the unet structure
# 2. figure out residual fusing path
#       - change h_deconv_concat = crop_and_concat(dw_h)convs[layer],h_deconv)
# 3. Try to implement 3 different paths connected by different weights
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd+'/../')

from tf_unet import unet, util, image_util
#preparing data loading
# modif_needed :: separate train/validation/test sets
# data_provider should include all image and label image
data_provider = image_util.ImageDataProvider("./data/pascalDB/*", n_class=21, data_suffix=".jpg",mask_suffix="_mask.png")
cwd = os.getcwd()
output_path = cwd + '/output/tutorial1/iter20_alpha1e-4/'

#setup & training
# input: 3 channels (RGB), features_root:#layer in the first layer, layers: # of layers
net = unet.Unet(layers=5, features_root=64, channels=3, n_class=21)
trainer = unet.Trainer(net,optimizer="adam", opt_kwargs=dict(learning_rate=1e-4))
path = trainer.train(data_provider, output_path, training_iters=20, epochs=100)


test_data_provider = image_util.ImageDataProvider("./data/testDB/*", n_class=21, data_suffix=".jpg", mask_suffix="_mask.png")
test_data, test_label =
path = cwd + '/output/tutorial1/'
prediction = net.predict(path, data)
unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
img = util.combine_img_prediction(data, label, prediction)
util.save_image(img, "prediction.jpg")
