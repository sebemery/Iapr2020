import sys
sys.path.append('..')
from helpers import *
from robot_tracking import *
from element_location import *
from operator_classifier.operators_classifier import classify_operator
import matplotlib.pyplot as plt
import numpy as np
import argparse 
import matplotlib.patches as mpatches
import math 
from digits_classifier.CNN import ConvNet
import torch
import os 
import pickle
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage.color import rgb2gray

# parser to take arguments from command line
parser = argparse.ArgumentParser(description='IAPR project')

parser.add_argument('--input', type = str, 
                    default = '../data/robot_parcours_1.avi', help = 'path for input video')

parser.add_argument('--output', type = str, 
                    default = '../output/video.avi', help = 'path for output video')

args = parser.parse_args()

# print arguments given in command lines and 
print('input path :', args.input)
print('output path :', args.output)
output_path_directory = os.path.dirname(args.output)+'/'

# create output directory
if not os.path.exists(output_path_directory):
    os.makedirs(output_path_directory)

##############################################################################################
## Load data

# extract images from video
extract_images(data_path=args.input)

# load data from extracted images
input_images, input_images_names = load_data(plot_images=False)

##############################################################################################
## Main loop

print('> Processing images')

positions = []
operation_order = []
label_order = []
order = 0
result = 0

# model for digits recognition
model = ConvNet()
model.load_state_dict(torch.load('../digits_classifier/model/digits_classifier.pth'))
model.eval()

# model for operators recognition
classes = {'addition':'+', 'equal':'=', 'substraction':'-', 'division':'/', 'multiplication':'*'}
model_operator = pickle.load(open('../operator_classifier/model/model_operators.sav', 'rb'))

#Main Loop
for ind, im in enumerate(input_images): 
    
    # get object positions from first image
    if ind == 0: 
        arrow_bb = get_arrow_bb(input_images[0])
        segmented = segmentation(input_images[0], arrow_bb)
        boxes, last_ind = get_object_bb(segmented)
        checkboxes(boxes, last_ind)
    
    fig, ax = plt.subplots()
    ax.imshow(im)
    ax.axis('off') 
    
    # Robot bb
    arrow_bb = get_arrow_bb(im)
    minr_a, minc_a, maxr_a, maxc_a = arrow_bb
    x_robot, y_robot = minc_a+(maxc_a-minc_a)/2, minr_a+(maxr_a-minr_a)/2
    rect_a = mpatches.Rectangle((minc_a, minr_a), maxc_a - minc_a, maxr_a - minr_a, fill=False, edgecolor='red', linewidth=0.5)
    ax.add_patch(rect_a)
    
    # Object bb
    for box in boxes:
        minr, minc, maxr, maxc = boxes[box]
        x, y = minc+(maxc-minc)/2, minr+(maxr-minr)/2
        rect = mpatches.Rectangle((x-25, y-25), 50, 50, fill=False, edgecolor='blue', linewidth=0.5)
        
        # If Overlap btw boxes show bb
        if not Overlap([minc_a, minr_a], [maxc_a, maxr_a], [x-25, y-25], [x+25, y+25]):
            ax.add_patch(rect)

        # If robot overlaps label bb crop image and classify what's inside
        if Overlap([minc_a+10, minr_a+10], [maxc_a-10, maxr_a-10], [x-10, y-10], [x+10, y+10]):
            if box not in label_order:
                
                # crop image 
                crop_sizex = int((maxc-minc)*0.80)
                crop_sizey = int((maxr-minr)*0.80)
                crop_size = max(crop_sizex, crop_sizey)
                crop_im = crop_image(input_images[0], [int(x-crop_size), int(y-crop_size), int(x+crop_size), int(y+crop_size)])
                
                # even -> classify digit
                if order%2 == 0: 
                    digit = resize(crop_im, (40,40), anti_aliasing = True)
                    digit = rgb2gray(digit)
                    thresh = threshold_otsu(digit)
                    digit = digit<thresh
                    digit_test = torch.from_numpy(digit.astype(np.float32)).reshape(1,1,40,40) 
                    output_test = model(digit_test).softmax(1)
                    _, predicted_classe = torch.max(output_test, 1)
                    predicted_classe = predicted_classe.item()
                    order += 1
                
                # odd -> classify operators 
                else: 
                    predicted_classe = classes[classify_operator(crop_im, model_operator)]
                    order += 1

                label_order.append(box)
                operation_order.append(predicted_classe)
            
    # Robot positions
    if len(positions) > 1:
        for last, pos in zip(positions,positions[1:]):
            point = mpatches.Circle(pos, 2, color='r')
            ax.add_patch(point)
            ax.plot((last[0], pos[0]), (last[1], pos[1]), '-k', linewidth=0.5)
    
    if ind > 0:
        positions.append((x_robot, y_robot))
    
    # Operation order display
    string = ''
    if operation_order:
        for op in operation_order:
            string += str(op)
            if op == '=':
                try:
                    result = eval(string[:len(string)-1])
                    string += str(result)
                except (ValueError, SyntaxError):
                    string += 'FALSE'
                    
    ax.annotate(string, (75, 400), color='g', fontsize=10)

    if ind < 10:
        plt.savefig('../output/frameout0'+str(ind)+'.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    else: 
        plt.savefig('../output/frameout'+str(ind)+'.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

print('done')

##############################################################################################
## Create output video from processed images

print('> Creating output video from processed images')

fps = 6
frames_to_video(output_path_directory, args.output, fps)
print('done')
try:
    print('The equation is the following : '+string)
except (ValueError, SyntaxError):
    print('The equation is the following : FALSE')
