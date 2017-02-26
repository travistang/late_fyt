from gym_torcs import TorcsEnv
import numpy as np
import random
import matplotlib.pyplot as plt
import caffe,cv2
img = None
def playGame():    #1 means Train, 0 means simply Run
    caffe.set_mode_gpu()
    segnet = caffe.Net('segnet_basic_inference.prototxt','segnet_pretrained.caffemodel',caffe.TEST)
    net = caffe.Net('torcs_net.prototxt','/home/travis/Desktop/segnet-yolo/trial12/weights2/Combined_iter_40000.caffemodel',caffe.TEST)
    # Generate a Torcs environment
    env = TorcsEnv(vision=True, throttle=False,gear_change=False)
    
    decision = np.random.randn(2)
    for j in range(10000000):
        ob, r_t, done, info = env.step(decision)
        vision = ob[-1]
        img = np.ndarray((64,64,3))
        for i in range(3):
            img[:, :, i] = 255 - vision[:, i].reshape((64, 64))
        img = np.flipud(img)
        img = cv2.resize(img,(227,227))
        img = img.transpose(2,0,1)
        segnet.blobs['data'].data[...] = img
        res = segnet.forward()
        segnet_out = res['segnet_out'].squeeze()
        cv2.imwrite('%d.png' % j,segnet_out)
        segnet_out = cv2.resize(segnet_out,(64,64))
        edges = cv2.Canny(np.array(segnet_out,dtype = np.uint8),10,20)
        net.blobs['data'].data[...] = edges
        res = net.forward()
        decision[1] = res['throttle_out']
        decision[0] = res['steer_out']
        print decision
    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
