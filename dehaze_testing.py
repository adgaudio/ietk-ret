"""
Some plots of dehazing and color illumination
"""
import random
import multiprocessing as mp
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from ietk.methods.dehaze import *

if __name__ == "__main__":
    fps_healthy = glob.glob('./data/messidor_healthy/*/*')
    fps_grade3 = glob.glob('./data/messidor_grade3/*/*')

    test1 = True
    test2 = False
    test3 = False

    if test1:
        # Test 1.  compare illumination correction vs dehaze->illuminate
        def illuminate_and_dehaze_test(fp):
            """Dehaze and illumination experiment"""
            _dehaze_dct = dehaze_from_fp(fp)

            img = _dehaze_dct['img']
            illuminated = illumination_correction(img.copy())['radiance']
            illuminated_dehazed = illuminate_from_fp(fp)[1]['radiance']
            dehazed = _dehaze_dct['radiance']
            dehazed_illuminated = illumination_correction(dehazed.copy())['radiance']
            return locals()

        grid_shape = (2, 2)
        #  imgs = [U.tonp(U.read_img(fp)) for fp in fps_healthy[:grid_shape[0]]]
        fps_shuffled = [x for x in fps_healthy]
        random.shuffle(fps_shuffled)
        #  imgs = [U.tonp(U.read_img(fp)) for fp in fps_shuffled]

        def norm01(img):
            """make image presentable for plotting"""
            return img / img.max()
        with mp.Pool(3) as pool:
            results = pool.imap(illuminate_and_dehaze_test, fps_shuffled[:10])
            while True:
                #  fig = plt.figure(1)
                #  grid = ImageGrid(fig, 111, grid_shape)
                fig, axs = plt.subplots(2, 2, num=1)
                grid = axs.ravel()
                #  results = [illuminate_and_dehaze_test(fps_healthy[1])]
                dct = next(results)

                axs = plt.subplots(1, 2, num=2)[1]
                axs[0].imshow(dct['img'])
                z = dct['illuminated_dehazed']
                #  z = (z - z.min((0,1), keepdims=True))/ (z.max((0,1), keepdims=True) - z.min((0,1), keepdims=True))
                axs[1].imshow(z/z.max((0,1), keepdims=True))
                #  grid[grid_shape[1]*n].imshow(dct['img'])
                grid[0].imshow(norm01(dct['illuminated']))
                grid[0].set_title('illuminated')
                grid[1].imshow(norm01(dct['dehazed']))
                grid[1].set_title('dehazed')
                grid[2].imshow(norm01(dct['illuminated_dehazed']))
                grid[2].set_title('illuminated and then dehazed')
                grid[3].imshow(norm01(dct['dehazed_illuminated']))
                grid[3].set_title('dehazed and then illuminated')
                plt.show()


    # Test 2: show the effect of dehazing
    if test2:
        def norm01(img):
            """make image presentable for plotting"""
            return img / img.max()
        #  ImageGrid
        grid_shape = (3, 4)
        #  imgs = [U.tonp(U.read_img(fp)) for fp in fps_healthy[:grid_shape[0]]]
        fps_shuffled = [x for x in fps_grade3]
        random.shuffle(fps_shuffled)
        #  imgs = [U.tonp(U.read_img(fp)) for fp in fps_shuffled]

        with mp.Pool(3) as pool:
            results = pool.imap(illuminate_from_fp, fps_shuffled)
            while True:
                fig = plt.figure(1)
                grid = ImageGrid(fig, 111, grid_shape)
                for n, dct in enumerate(results):
                    grid[grid_shape[1]*n].imshow(dct[1]['img'])
                    grid[grid_shape[1]*n+1].imshow(norm01(dct[1]['t_unrefined']), cmap='Greys')
                    grid[grid_shape[1]*n+2].imshow(norm01(dct[1]['t_refined']), cmap='Greys')
                    grid[grid_shape[1]*n+3].imshow(norm01(dct[1]['radiance']))
                    if n%grid_shape[0] == grid_shape[0]-1:
                        break
                f, axs = plt.subplots(1, 2)
                axs[0].imshow(dct[1]['img'])
                axs[1].imshow(dct[1]['radiance'])
                plt.show()

    # Test 3: show the effect of illuminate->dehaze
    if test3:
        def norm01(img):
            """make image presentable for plotting"""
            return img#(img -img.min())/ (img.max()-img.min())
        #  ImageGrid
        #  imgs = [U.tonp(U.read_img(fp)) for fp in fps_healthy[:grid_shape[0]]]
        fps_shuffled = [x for x in fps_grade3]
        random.shuffle(fps_shuffled)
        #  imgs = [U.tonp(U.read_img(fp)) for fp in fps_shuffled]

        with mp.Pool(3) as pool:
            results = pool.imap(illuminate_from_fp, fps_shuffled[:10])
            while True:
                fig = plt.figure(1)
                grid_shape = (3, 5)
                grid = ImageGrid(fig, 111, grid_shape)
                for n, (ill, deh) in enumerate(results):
                    grid[grid_shape[1]*n+0].imshow(norm01(ill['img']), cmap='Greys')
                    grid[grid_shape[1]*n+1].imshow(norm01(ill['t_refined']), cmap='Greys')
                    grid[grid_shape[1]*n+2].imshow(norm01(ill['radiance']), cmap='Greys')
                    grid[grid_shape[1]*n+3].imshow(norm01(deh['t_refined']), cmap='Greys')
                    grid[grid_shape[1]*n+4].imshow(norm01(deh['radiance']), cmap='Greys')
                    if n%grid_shape[0] == grid_shape[0]-1:
                        break
                plt.show()
