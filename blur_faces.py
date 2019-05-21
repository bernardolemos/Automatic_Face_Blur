import os
import cv2
import time
import argparse
import numpy as np
from mtcnn import detect_face
import tensorflow as tf
from PIL import Image, ImageDraw

## MTCNN face localizer
def mtcnn_localize_faces(image, pnet, rnet, onet, minsize=20, threshold=[0.7, 0.8, 0.85], factor=0.75):
    """
    Localize faces & its landmarks in image using MTCNN
    
    Params
    :image
    :minsize - min. face size
    :threshold - a list/array with 3 values. The thresholds for pnet, rnet & onet, respectively 
    :factor - sclaing factor for image octave

    Return
    :bbs - list of bounding boxes
    :lds - list of face landmarks
    """
    

    image = image[:, :, 0:3]
    bounding_boxes, landmarks = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]

    bbs = list()
    lds = list()
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
                
        bb = np.zeros((nrof_faces,4), dtype=np.int32)
        lands = np.zeros((nrof_faces,10), dtype=np.int32)
        landmarks = np.reshape(landmarks, (nrof_faces, 10))
        for i in range(nrof_faces):
            ## Convert to int32
            lands[i] = np.ravel(landmarks[i])
            bb[i] = np.ravel(det[i])
            # inner exception
            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(image[0]) or bb[i][3] >= len(image):
                print('face is inner of range!')
                continue
            else:
                ## get as top, right, bottom, left
                bbs.append((bb[i][1], bb[i][2], bb[i][3], bb[i][0]))
                lds.append(lands[i])
                
    return bbs, lds


def load_images(images_path):
    """
    Read images from directory

    Params
    :images_path - path to images

    Return
    :image_l - list of images as arrays
    : images_name - list of images' file names
    """
    # list of images, as arrays
    images_l = []
    # get images
    images_name = os.listdir(images_path)
    # read images
    for i in images_name:
        image = cv2.imread(os.path.join(images_path, i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if image.endswith(".png"):
        #     images_l.append(image)
        images_l.append(image)
    
    return images_l, images_name

def main(args):
    st = time.time()
    #check if input directory exists
    if not os.path.exists(args.input_directory):
        print("Error! No input direcotory", args.input_directory)
        return -1

    # read images
    images_l, images_paths = load_images(args.input_directory)

    #create tensorflow session
    # init. tensorflow session
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './mtcnn')
            #localize and blur faces, iterate over images
            for image, image_path in zip(images_l, images_paths):
                print("Processing", image_path + "...")

                bbs, lds = mtcnn_localize_faces(image, pnet, rnet, onet, minsize=20, threshold=[0.7, 0.8, 0.85], factor=0.75)

                # jumpt iteration if there's no face
                if len(bbs) == 0:
                    print("Couldn't find faces!")
                    continue

                #get faces
                for bb, ld in zip(bbs, lds):
                    #get bounding box
                    #top, righ, bottom, left
                    top = bb[0]
                    right = bb[1]
                    bottom = bb[2]
                    left = bb[3]
                    # build landmarks' x, y pairs
                    points = []
                    for x, y in zip(ld[:5], ld[5:]):
                        points.append(x)
                        points.append(y)

                    #get face thumbnail
                    face_image = image[top:bottom, left:right]
                    #blur face thumbnail
                    if args.blur > 0:
                        face_image = cv2.GaussianBlur(face_image, (105, 105), args.blur)
                    #black
                    else:
                        face_image = np.zeros(face_image.shape)
                    
                    #write blured face to image
                    image[top:bottom, left:right] = face_image

                    #PIL image 
                    # pil_image = Image.fromarray(image)
                    # pil_image_face = Image.fromarray(face_image)

                    #eyes' landmarks: first two pairs
                    # get larger rectangle
                    # points[0] = points[0] * 0.9
                    # points[1] = points[1] * 0.9
                    # points[2] = points[2] * 1.1
                    # points[3] = points[3] * 1.1
                    # draw = ImageDraw.Draw(pil_image)
                    #cover eyes with rectangle
                    # draw.rectangle(points[:4], fill="black")

                #create output directory if it doesn't exist
                if not os.path.exists(args.output_directory):
                    os.makedirs(args.output_directory)

                #save image
                pil_image = Image.fromarray(image)
                pil_image.save(os.path.join(args.output_directory, image_path))

    print("Total running time:", time.time() - st, "sec.")
    
    return 0

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_directory', type=str, nargs='?', default="./images")
    parser.add_argument('-od', '--output_directory', type=str, nargs='?', default="./blurs")
    parser.add_argument('-b', '--blur', type=int, nargs='?', default=46)
    args = parser.parse_args()

    main(args)