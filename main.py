from tensorflow.keras import backend as K
from process import preprocess_image, deprocess_image
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time
from evaluator import Evaluator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--target", required=True,
        help="path to TARGET IMAGE")
ap.add_argument("-s", "--style", required=False,
        help="path to  STYLE IMAGE")

args = vars(ap.parse_args())



target_image_path = args['target']
style_reference_image_path = args['style']

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height/ height)

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis = 0)

model = vgg19.VGG19(input_tensor = input_tensor,
                    weights= 'imagenet',
                    include_top = False)

print('Model Loaded')

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
total_variation_weight = 10 **  -4 
style_weight = 1. 
content_weight = 0.025 


loss = K.variable(0.)
layer_features = outputs_dict[content_layer]

target_image_features = layer_features[0, :, :, :]

combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,combination_features)
for layer_name in style_layers:

  layer_features = outputs_dict[layer_name]

  style_reference_features = layer_features[1, :, :, :]

  combination_features = layer_features[2, :, :, :]
  sl = style_loss(style_reference_features, combination_features)
  loss += (style_weight / len(style_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

evaluator = Evaluator()

result_prefix = 'my_result'
iterations = 20

x = preprocess_image(target_image_path)
x = x.flatten()

for i in range(iterations):
  print('Start of iteration', i)

  start_time = time.time()
  x, min_val, info = fmin_l_bfgs_b(evaluator.loss,x, fprime=evaluator.grads,maxfun=20)

  print('Current loss value:', min_val)

  img = x.copy().reshape((img_height, img_width, 3))
  img = deprocess_image(img)
  fname = result_prefix + '_at_iteration_%d.png' % i

  imsave(fname, img)

  print('Image saved as', fname)
  end_time = time.time()
  print('Iteration %d completed in %ds' % (i, end_time - start_time))