
# coding: utf-8

# In[30]:


import os
import sys
import torch
import dlib
from PIL import Image,ImageOps
import numpy as np
import cv2
from skimage import feature
import random
import torchvision.transforms as transforms
from keypoint2img import interpPoints,drawEdge


# In[2]:


# general variable

dir_A='./train_keypoints/'
dir_B='./train_img/'

loadSize=256

scale_ratio = np.array(
            [[0.9, 1], [1, 1], [0.9, 1], [1, 1.1], [0.9, 0.9], [0.9, 0.9]]) 
scale_ratio_sym = np.array(
            [[1, 1], [0.9, 1], [1, 1], [0.9, 1], [1, 1], [1, 1]])
scale_shift = np.zeros((6, 2))

min_x=0.0
min_y=0.0
max_x=0.0
max_y=0.0

# whether random scale points 
random_scale_points=True


# In[3]:


# image folder process

def make_grouped_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    fnames = sorted(os.walk(dir))
    for fname in sorted(fnames):
        paths = []
        root = fname[0]
        for f in sorted(fname[2]):
            if is_image_file(f):
                paths.append(os.path.join(root, f))
        if len(paths) > 0:
            images.append(paths)
    return images


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tiff', '.txt', '.json'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def check_path_valid(A_paths, B_paths):
    assert (len(A_paths) == len(B_paths))
    for a, b in zip(A_paths, B_paths):
        assert (len(a) == len(b))


# In[4]:


# 对面部的处理写成一个类会好一些。
class FaceDataset():
    def initialize(self):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.A_paths=sorted(make_grouped_dataset(dir_A))
        self.B_paths=sorted(make_grouped_dataset(dir_B))
        check_path_valid(A_paths,B_paths)
        self.scale_ratio = np.array(
            [[0.9, 1], [1, 1], [0.9, 1], [1, 1.1], [0.9, 0.9], [0.9, 0.9]])  # np.random.uniform(0.9, 1.1, size=[6, 2])
        self.scale_ratio_sym = np.array(
            [[1, 1], [0.9, 1], [1, 1], [0.9, 1], [1, 1], [1, 1]])  # np.random.uniform(0.9, 1.1, size=[6, 2])
        self.scale_shift = np.zeros((6, 2))  # np.random.uniform(-5, 5, size=[6, 2])
        
    def __getitem__(self, index):
        pass
        


# In[5]:


# basic proess for dataset

def get_img_params(size):
    w, h = size
    new_h = new_w = loadSize
    # scale width
    new_w = loadSize
    new_h = loadSize * h // w
    new_w = int(round(new_w / 4)) * 4
    new_h = int(round(new_h / 4)) * 4
    flip=(random.random()>0.5) 
    return {'new_size': (new_w,new_h),'flip': flip}

def get_transform(params, method=Image.BICUBIC,normalize=True):
    transform_list=[]
    osize = [loadSize, loadSize]
    transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))
    transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size    
    return img.resize((w, h), method)

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


# In[6]:


# image process ---crop, transform    
    
def get_crop_coords(keypoints, size):
        min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()
        min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
        offset = (max_x - min_x) // 2
        min_y = max(0, min_y - offset * 2)
        min_x = max(0, min_x - offset)
        max_x = min(size[0], max_x + offset)
        max_y = min(size[1], max_y + offset)
        return {'min_y':int(min_y),'max_y':int(max_y),'min_x':int(min_x),'max_x':int(max_x)}
#         return min_y, max_y, min_x, max_x = int(min_y), int(max_y), int(min_x), int(max_x)

def crop(img):
    if isinstance(img, np.ndarray):
        return img[min_y:max_y, min_x:max_x]
    else:
        return img.crop((min_x, min_y, max_x, max_y)) 

def scale_points(keypoints, part, index, sym=False):
    if sym:
        pts_idx = sum([list(idx) for idx in part], [])
        pts = keypoints[pts_idx]
        ratio_x = scale_ratio_sym[index, 0]
        ratio_y = scale_ratio_sym[index, 1]
        mean = np.mean(pts, axis=0)
        mean_x, mean_y = mean[0], mean[1]
        for idx in part:
            pts_i = keypoints[idx]
            mean_i = np.mean(pts_i, axis=0)
            mean_ix, mean_iy = mean_i[0], mean_i[1]
            new_mean_ix = (mean_ix - mean_x) * ratio_x + mean_x
            new_mean_iy = (mean_iy - mean_y) * ratio_y + mean_y
            pts_i[:,0] = (pts_i[:,0] - mean_ix) + new_mean_ix
            pts_i[:,1] = (pts_i[:,1] - mean_iy) + new_mean_iy
            keypoints[idx] = pts_i

    else:            
        pts_idx = sum([list(idx) for idx in part], [])
        pts = keypoints[pts_idx]
        ratio_x = scale_ratio[index, 0]
        ratio_y = scale_ratio[index, 1]
        mean = np.mean(pts, axis=0)
        mean_x, mean_y = mean[0], mean[1]            
        pts[:,0] = (pts[:,0] - mean_x) * ratio_x + mean_x + scale_shift[index, 0]
        pts[:,1] = (pts[:,1] - mean_y) * ratio_y + mean_y + scale_shift[index, 1]
        keypoints[pts_idx] = pts


# In[58]:


# Test scale_points 
part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]],  # face
                     [range(17, 22)],  # right eyebrow
                     [range(22, 27)],  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],  # nose
                     [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
                     [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
                     [range(48, 55), [54, 55, 56, 57, 58, 59, 48]],  # mouth
                     [range(60, 65), [64, 65, 66, 67, 60]]  # tongue
                     ]
keypoints = np.loadtxt(A_path, delimiter=',')
# print('kp0:',keypoints[:-5])
scale_points(keypoints, part_list[1] + part_list[2], 1, sym=True)
# print('kp1:',keypoints[:-5])


# In[55]:


# face region extraction and get edge maps
def get_image(A_path, transform_scaleA):
    A_img = Image.open(A_path)                
    A_scaled = transform_scaleA(crop(A_img))
    return A_scaled
    
def get_face_image(A_path, transform_A, transform_L, size, img):
    # read face keypoints from path and crop face region
    keypoints, part_list, part_labels = read_keypoints(A_path, size)

    # draw edges and possibly add distance transform maps
    add_dist_map = False
    im_edges, dist_tensor = draw_face_edges(keypoints, part_list, transform_A, size, add_dist_map)

    # canny edge for background
    no_canny_edge = False
    if not no_canny_edge:
        edges = feature.canny(np.array(img.convert('L')))        
        edges = edges * (part_labels == 0)  # remove edges within face
        im_edges += (edges * 255).astype(np.uint8)
    im_edges= ImageOps.invert(Image.fromarray(im_edges).convert('RGB'))
    im_edges.show()
    im_edges = np.array(im_edges.convert('L'))
    edge_tensor = transform_A(Image.fromarray(crop(im_edges)))
    edge_tensor = torch.from_numpy(np.array(edge_tensor))
    # final input tensor
    input_tensor = torch.cat([edge_tensor, dist_tensor]) if add_dist_map else edge_tensor
    label_tensor = torch.from_numpy(np.array(transform_L(Image.fromarray(crop(part_labels.astype(np.uint8)))))) * 255.0
    return input_tensor, label_tensor

def read_keypoints(A_path, size):
    # mapping from keypoints to face part
    part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]],  # face
                 [range(17, 22)],  # right eyebrow
                 [range(22, 27)],  # left eyebrow
                 [[28, 31], range(31, 36), [35, 28]],  # nose
                 [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
                 [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
                 [range(48, 55), [54, 55, 56, 57, 58, 59, 48]],  # mouth
                 [range(60, 65), [64, 65, 66, 67, 60]]  # tongue
                 ]
    label_list = [1, 2, 2, 3, 4, 4, 5, 6]  # labeling for different facial parts
    keypoints = np.loadtxt(A_path, delimiter=',')
    print(keypoints.shape)
    # add upper half face by symmetry
    pts = keypoints[:17, :].astype(np.int32)
    baseline_y = (pts[0, 1] + pts[-1, 1]) / 2
    upper_pts = pts[1:-1, :].copy()
    upper_pts[:, 1] = baseline_y + (baseline_y - upper_pts[:, 1]) * 2 // 3
    keypoints = np.vstack((keypoints, upper_pts[::-1, :]))

    # label map for facial part
    w, h = size
    part_labels = np.zeros((h, w), np.uint8)
    for p, edge_list in enumerate(part_list):
        indices = [item for sublist in edge_list for item in sublist]
        pts = keypoints[indices, :].astype(np.int32)
        cv2.fillPoly(part_labels, pts=[pts], color=label_list[p])

    # move the keypoints a bit
    if random_scale_points:
        scale_points(keypoints, part_list[1] + part_list[2], 1, sym=True)
        scale_points(keypoints, part_list[4] + part_list[5], 3, sym=True)
        for i, part in enumerate(part_list):
            scale_points(keypoints, part, label_list[i] - 1)

    return keypoints, part_list, part_labels

def draw_face_edges(keypoints, part_list, transform_A, size, add_dist_map):
    w, h = size
    edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
    # edge map for face region from keypoints
#     im_edges = np.zeros((h, w), np.uint8)  # edge map for all edges
    im_edges = np.zeros((h,w),np.uint8)
    dist_tensor = 0
    e = 1
    for edge_list in part_list:
#         print('edge_list',edge_list)
        for edge in edge_list:
#             print('edge',edge)
#             im_edge = np.zeros((h, w), np.uint8)  # edge map for the current edge
            im_edge = np.zeros((h,w),np.uint8)
            for i in range(0, max(1, len(edge) - 1),
                           edge_len - 1):  # divide a long edge into multiple small edges when drawing
                sub_edge = edge[i:i + edge_len]
                x = keypoints[sub_edge, 0]
                y = keypoints[sub_edge, 1]
                curve_x, curve_y = interpPoints(x, y)  # interp keypoints to get the curve shape
                drawEdge(im_edges, curve_x, curve_y)
#                 im_edges=Image.fromarray(im_edges)
#                 im_edges.show()
#                 im_edges=np.array(im_edges)
                if add_dist_map:
                    drawEdge(im_edge, curve_x, curve_y)

            if add_dist_map:  # add distance transform map on each facial part
                im_dist = cv2.distanceTransform(255 - im_edge, cv2.DIST_L1, 3)
                im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
                im_dist = Image.fromarray(im_dist)
                tensor_cropped = transform_A(crop(im_dist))
                tensor_cropped = np.array(tensor_cropped)
                tensor_cropped = torch.from_numpy(tensor_cropped)
                dist_tensor = tensor_cropped if e == 1 else torch.cat([dist_tensor, tensor_cropped])
                e += 1

    return im_edges, dist_tensor


# In[56]:


# Test
def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():            
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]            
        else:            
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]

def drawEdge(im, x, y, bw=1, color=(255,255,255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)
#         print('im changed',im[yy,xx])
        # edge endpoints
        if draw_end_points:
            for i in range(-bw*2, bw*2):
                for j in range(-bw*2, bw*2):
                    if (i**2) + (j**2) < (4 * bw**2):
                        yy = np.maximum(0, np.minimum(h-1, np.array([y[0], y[-1]])+i))
                        xx = np.maximum(0, np.minimum(w-1, np.array([x[0], x[-1]])+j))
                        setColor(im, yy, xx, color)


# In[57]:


# Test and get targets
A_paths=sorted(make_grouped_dataset(dir_A))
B_paths=sorted(make_grouped_dataset(dir_B))
check_path_valid(A_paths,B_paths)

for i,files in enumerate(B_paths):
    for index, image_name in enumerate(files):
        if i==0 and index==0:
            A_path=A_paths[i][index]
            B_path=B_paths[i][index]
            B_img = Image.open(B_path).convert('RGB') 
#             B_img.show()            
            B_size = B_img.size
            points = np.loadtxt(A_path, delimiter=',')
#             print("B size,",B_size)
#             print(points)
            value = get_crop_coords(points,B_size)
            min_y = value['min_y']
            max_y = value['max_y']
            min_x = value['min_x']
            max_x = value['max_x']
#             print("min x,max x; min y, max y",min_x,max_x,min_y,max_y)
            params=get_img_params(crop(B_img).size)
            transforms_saleA = get_transform(params, method=Image.BILINEAR, normalize=False)
            transforms_label = get_transform(params, method=Image.NEAREST, normalize=False)
            transforms_scaleB = get_transform(params,normalize=False)
            A,L = get_face_image(A_path, transforms_saleA, transforms_label, B_size, B_img)
#             B = transforms_scaleB(torch.from_numpy((crop(np.array(B_img)))))
            B = transforms_scaleB(crop(B_img))
            A = Image.fromarray(A.numpy())
            A.show()
#             B=Image.fromarray(B.numpy())
            B.show()
            L=Image.fromarray(L.numpy())
            L.show()


# In[ ]:


a=torch.randn(3,3)
print(a)


# In[ ]:


b=torch.randn((3,3))
print(b)

