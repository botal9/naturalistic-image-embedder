from naturalistic_image_embedder.common.image import insert_image, InsertionType

import os
import sys
import webbrowser


def run(background_image_path,
        foreground_image_path,
        out_image_path,
        offset_x,
        offset_y,
        insertion_type_name):
    if insertion_type_name == 'color_transfer':
        insertion_type = InsertionType.COLOR_TRANSFER
    elif insertion_type_name == 'poisson_blending':
        insertion_type = InsertionType.POISSON_BLENDING_MIXED
    elif insertion_type_name == 'naive':
        insertion_type = InsertionType.NAIVE
    else:
        print(f'Incorrect insertion type: "{insertion_type_name}"')
        return

    insert_image(background_image_path,
                 foreground_image_path,
                 out_image_path,
                 (int(offset_x), int(offset_y)),
                 insertion_type)


def run_sample():
    offsets = [(20, 210), (1000, 800), (0, 0), (1100, 850)]
    proj_dir = os.path.split(__file__)[0]
    image_dir = os.path.join(proj_dir, 'images')
    out_dir = os.path.join(proj_dir, 'out')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    images = sorted(os.listdir(image_dir))
    for i in range(len(images) // 2):
        image_no = i + 1
        background_image_path = os.path.join(image_dir, images[2 * i])
        foreground_image_path = os.path.join(image_dir, images[2 * i + 1])

        for insertion_type in InsertionType:
            insertion_type_name = insertion_type.name.lower()
            out_image_path = os.path.join(out_dir, f'{image_no}_{insertion_type_name}.jpg')
            insert_image(background_image_path,
                         foreground_image_path,
                         out_image_path,
                         offsets[i],
                         insertion_type)


EVALUATE_REALISM = False
REALISM_CNN_FOLDER = '/home/ovechkinve/itmo/diploma/RealismCNN'


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_sample()
        if EVALUATE_REALISM:
            script_path = os.path.join(REALISM_CNN_FOLDER, 'PredictRealism.m')
            html_path = os.path.join(REALISM_CNN_FOLDER, 'web/ranking/realism_cnn_all/index.html')
            os.system(f'matlab -batch "run(\'{script_path}\');exit;" >/dev/null 2>&1')
            webbrowser.open_new(html_path)
    else:
        if len(sys.argv) != 7:
            print('Need to specify 6 arguments: background, foreground and out image paths, '
                  'insertion offsets (X and Y coords) and insertion type which is one of '
                  '"color_transfer", "naive", "poisson_blending".')
        else:
            run(*sys.argv[1:])
