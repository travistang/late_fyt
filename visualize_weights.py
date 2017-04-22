from keras.models import *


net = model_from_json('actormodel.json')
net.load_weights('actormodel.h5')
