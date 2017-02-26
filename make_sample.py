from gym_torcs import TorcsEnv
import numpy as np
import cv2
PI= 3.14159265359

def extract_images(vision):
    img = np.ndarray((64,64,3))
    for i in range(3):
        img[:, :, i] = 255 - vision[:, i].reshape((64, 64))
    img = np.flipud(img)
    img = img.transpose(2,0,1)
    return img.astype(np.float32)


env = TorcsEnv(vision = True,throttle = True, gear_change = False)
ob = env.reset()
max_steps = 10000000
action = [0,0.2,0] # steering, acceleration
for i in range(max_steps):
    vision = ob[-1]
    img = np.array(vision).reshape(64,64,3)
    img = cv2.cvtColor(np.flipud(img).astype(np.float32)/255.0,cv2.COLOR_BGR2RGB)
    np.save('images/%05d.npy' % i,img)
    #scene = extract_images(scene)
    ob,_,done,info = env.step(action)
    # cheat here
    c = env.get_client()
    S,R = c.S.d,c.R.d
    target_speed = 5
    if S[u'speedX'] < target_speed - (R[u'steer']*40):
        action[1]+= .01
    else:
        action[1]-= .01
    if S[u'speedX']<1:
        action[1] += 1/(S[u'speedX']+.1)
    # Traction Control System
    if ((S[u'wheelSpinVel'][2]+S[u'wheelSpinVel'][3]) - (S[u'wheelSpinVel'][0]+S[u'wheelSpinVel'][1]) > 5): action[1] -= .2
    # steering Control
    action[0] = S[u'angle']*3/PI
    action[0] -= S[u'trackPos']*.2

    if action[0] > 1: action[0] = 1
    if action[0] < -1: action[0] = -1
    # apply action
    env.step(np.array(action))

env.end()



'''
    focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel,_ = ob
    # acceleration control
    if speedX < target_speed - (R[u'steer']*50):
        action[1] += 0.1
    else:
        action[1] -= .01
#    if speedX<10:
#       action[1] += 1/(speedX + .1)
    if ((S[u'wheelSpinVel'][2]+S[u'wheelSpinVel'][3]) -
       (S[u'wheelSpinVel'][0]+S[u'wheelSpinVel'][1]) > 5):
       action[1] -= .2
    # check acceleration range
    if  0 > action[1]:
        action[1] = 0
    elif action[1] > 1:
        action[1] = 1
'''
