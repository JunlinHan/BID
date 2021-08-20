import numpy as np
import os
import cv2
import random
import numba as nb

# Set the parameters, this is the default parameters used in our paper.
WIDTH = 1024
HEIGHT = 512
CORES = os.cpu_count()
DEFAULT_N_BALLS = 150
num_balls_min = 100
num_balls_max = 300
alpha = 1
rain_min_size = 2
rain_max_size = 7
connected_metaball_num = 3


def update_balls(balls: np.ndarray, dt: float):
    # remove the ball outside the screen
    remove_index_list = []
    for b in range(balls.shape[0]):
        # move
        if((balls[b].pos[0]>WIDTH) or (balls[b].pos[1]>HEIGHT)):
            remove_index_list.append(b)

    balls = np.delete(balls,remove_index_list)

    # update position for large raindrops
    for b in range(balls.shape[0]):
        # move
        if(balls[b].radius>6):
            balls[b].pos += balls[b].vel * dt*0.2

    return balls


def gen_metaball():
    alpha = np.zeros((20, 20, 3))
    for i in range(20):
        for j in range(20):
            dx = (20 / 2 - i) / 10
            dy = (20 / 2 - j) / 10
            if ((dx != 0) or (dy != 0)):
                alpha[i, j, :] = 1 / (dx ** 2 + dy ** 2)
            else:
                alpha[i, j, :] = 4

    alpha[alpha > 4] = 4
    alpha = alpha / 4
    alpha[alpha < 0.25] = 0
    return alpha

# draw the textures image and alpha image for raindrop
def draw_textures(screen:np.ndarray,balls:np.ndarray,texture):
    w, h = screen.shape[0], screen.shape[1]
    b_count = balls.shape[0]

    # to use all cores
    for start in nb.prange(CORES):
        # for each pixel on screen
        for x in range(start, w, CORES):
            for y in range(h):
                screen[x, y].fill(0)  # clear pixel
                # for each ball

    # create texture screen
    texture_screen = screen.copy()

    for b in range(b_count):
        # print(b,balls[b].radius)
        # calculate value
        texture_w = min(w,int(balls[b].pos[0]+balls[b].radius*2))-max(0,int(balls[b].pos[0]-balls[b].radius*2))
        texture_h = min(h,int(balls[b].pos[1] + balls[b].radius * 2))-max(0,int(balls[b].pos[1] - balls[b].radius * 2))

        if((texture_w>0) and (texture_h>0)):

            alpha = gen_metaball()

            # resize to fit size
            alpha = cv2.resize(alpha, (texture_h, texture_w))
            texture = cv2.resize(texture,(texture_h,texture_w))


            screen[max(0, int(balls[b].pos[0] - balls[b].radius * 2)):min(w, int(
                balls[b].pos[0] + balls[b].radius * 2)),
            max(0, int(balls[b].pos[1] - balls[b].radius * 2)):min(h, int(
                balls[b].pos[1] + balls[b].radius * 2))] += \
                (alpha*255).astype(np.int32)


            texture_screen[max(0,int(balls[b].pos[0]-balls[b].radius*2)):min(w,int(balls[b].pos[0]+balls[b].radius*2)),
            max(0,int(balls[b].pos[1] - balls[b].radius * 2)):min(h,int(balls[b].pos[1] + balls[b].radius * 2))] += \
                (texture * screen[max(0,int(balls[b].pos[0]-balls[b].radius*2)):min(w,int(balls[b].pos[0]+balls[b].radius*2)),
            max(0,int(balls[b].pos[1] - balls[b].radius * 2)):min(h,int(balls[b].pos[1] + balls[b].radius * 2))] / 255 ).astype(np.int32)

            texture_screen[max(0,int(balls[b].pos[0]-balls[b].radius*2)):min(w,int(balls[b].pos[0]+balls[b].radius*2)),
            max(0,int(balls[b].pos[1] - balls[b].radius * 2)):min(h,int(balls[b].pos[1] + balls[b].radius * 2)),0] = balls[b].thickness*255


            texture_screen[texture_screen>255] = 255
            screen[texture_screen > 255] = 255

    return texture_screen,screen.copy()


def create_balls(n_balls):
    """make random balls"""
    balls = np.recarray(
        (n_balls,), dtype=[("pos", ("<f4", (2,))), ("radius", int),("thickness", "f4"), ("vel", ("<f4", (2,)))],
    )

    i = 0
    while (True):
        if(i>=n_balls):
            break
        # generate ball
        balls[i].radius = random.randint(rain_min_size, rain_max_size)
        balls[i].pos = (
            np.random.randint(balls[i].radius, WIDTH - balls[i].radius),
            np.random.randint(balls[i].radius, HEIGHT - balls[i].radius),
        )
        # set thinckness
        balls[i].thickness = random.random()
        # set velocity, correlate with radius
        balls[i].vel = (0,balls[i].radius**2)
        # generate connected metaball as raindrops
        num = random.randint(1, connected_metaball_num)
        i = i + 1
        for j in range(1,num):
            if (i >= n_balls):
                break
            balls[i].radius = random.randint(rain_min_size, rain_max_size)
            balls[i].pos = (
                random.choice([
                np.random.randint(int(balls[i-j].pos[0]-1.8*balls[i-j].radius),int(balls[i-j].pos[0]-0.5*balls[i-j].radius)),
                    np.random.randint(int(balls[i-j].pos[0]+0.5*balls[i-j].radius),int(balls[i-j].pos[0]+1.8*balls[i-j].radius))]),
                random.choice([
                    np.random.randint(int(balls[i-j].pos[1] - 1.8 * balls[i-j].radius),
                                      int(balls[i-j].pos[1] - 0.5 * balls[i-j].radius)),
                    np.random.randint(int(balls[i-j].pos[1] + 0.5 * balls[i-j].radius),
                                      int(balls[i-j].pos[1] + 1.8 * balls[i-j].radius))])
            )
            balls[i].thickness = random.random()
            # set the velocity same as the parent
            balls[i].vel = (0, balls[i].radius ** 2)
            i = i+1


    return balls

# add new balls
def add_balls(old_balls,n):

    balls = np.recarray(
        (n,), dtype=[("pos", ("<f4", (2,))), ("radius", int), ("thickness", "f4"),("vel", ("<f4", (2,)))],
    )
    for i in range(balls.shape[0]):
        # generate ball
        balls[i].radius = random.randint(rain_min_size, rain_max_size)

        balls[i].pos = (
            np.random.randint(balls[i].radius, WIDTH - balls[i].radius),
            np.random.randint(balls[i].radius, int(HEIGHT/2)),
        )
        balls[i].thickness = random.random()
        balls[i].vel = (0,balls[i].radius**2)

    new_balls = np.concatenate((old_balls,balls))

    return new_balls


def run():
    # set seed for repeatability
    screen = np.zeros((WIDTH, HEIGHT))
    # load the texture image for render
    texture = cv2.imread('../gen_raindrop/texture.png')

    counter = 0
    # How many images to generate.
    for i in range(4000):
        # necessary variables
        n_balls = random.randint(num_balls_min,num_balls_max)
        balls = create_balls(n_balls)

        # numpy array for screen
        screen_arr = np.zeros((WIDTH, HEIGHT, 3), dtype=np.int32)

        print(counter,'size:',balls.shape[0])
        counter +=1
        # numpy
        balls = add_balls(balls,50)

        # balls = update_balls(balls, dt)

        texture_screen,screen_ = draw_textures(screen_arr,balls,texture)
        texture_screen = np.array(texture_screen)

        # save images
        texture_screen = cv2.rotate(texture_screen,cv2.ROTATE_90_CLOCKWISE).astype('float32')
        texture_screen = cv2.resize(texture_screen,(int(WIDTH*alpha),HEIGHT))
        cv2.imwrite('alpha_textures/texture/texture_screen'+str(counter)+'.png',texture_screen)

        screen_ = cv2.rotate(screen_,cv2.ROTATE_90_CLOCKWISE).astype('float32')
        screen_ = cv2.resize(screen_, (int(WIDTH * alpha),HEIGHT))
        cv2.imwrite('alpha_textures/alpha/screen'+str(counter)+'.png',screen_)





if __name__ == "__main__":
    run()
