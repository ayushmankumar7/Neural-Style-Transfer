from keras import backend as K
from process import preprocess_image


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


