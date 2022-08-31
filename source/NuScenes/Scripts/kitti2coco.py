"""
    script to convert kitti-format output to coco-format
"""
import os
import numpy as np
import json
from PIL import Image

def load_paths(data_path):
  anno_files = os.listdir(data_path)
  print(anno_files)
  label_folder_path = '/content/OC_SORT/datasets/training/label_02/'
  image_folder_path = '/content/OC_SORT/datasets/training/image_02/00'
  image_files = []
  label_files = []
  for anno_f in anno_files:
    # print(anno_f)
    label_path = label_folder_path + anno_f
    image_folder = anno_f[2:4]
    image_files.append(image_folder_path + image_folder)
    print(image_folder_path + image_folder)
    label_files.append(label_path)
  return image_files, label_files    

if __name__ == "__main__":
    src_path, out_path = '/content/OC_SORT/datasets/training/label_02/', '/content/OC_SORT/datasets/training/train.json'  #Pass the label_02 folder path and out_path as args
    out_path_json = out_path + 'train.json'
    out = {'images': [], 'annotations': [], 'categories': [{"id" : 0, "name" : "DontCare", "supercategory" : None}, {"id" : 1, "name" : "Van", "supercategory" : None}, {"id" : 2, "name" : "Car", "supercategory" : None}, {"id" : 3, "name" : "Pedestrian", "supercategory" : None}, {"id" : 4, "name" : "Cyclist", "supercategory" : None}]}
    img_folder_paths, label_paths = load_paths(src_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0 
    out_dict = {} 
    for img_folder_path, label_path in zip(img_folder_paths, label_paths):
        f = open(label_path)
        lines = f.readlines()
        terms = lines[0].split()
        prev_frame_id = terms[0]
        count = 0
        video_cnt += 1
        for i in range(len(lines)):
          line = lines[i]
          terms = line.split()
          if terms[0] == prev_frame_id and count == 0:
            count += 1
            image_cnt += 1
            if int(terms[0]) < 10:
              img_path = img_folder_path + "/00000" + str(terms[0]) + '.png'
            if int(terms[0]) < 100 and int(terms[0]) >= 10:
              img_path = img_folder_path + "/0000" + str(terms[0]) + '.png'
            if int(terms[0]) >= 100 and int(terms[0]) < 1000:
              img_path = img_folder_path + "/000" + str(terms[0]) + '.png'
            if int(terms[0]) >= 1000:
              img_path = img_folder_path + "/00" + str(terms[0]) + '.png'
            im = Image.open(img_path)
            image_info = {'file_name': img_path, 'id': image_cnt, 'filename' : img_path, 'height': im.size[1], 'width': im.size[0], 'video_id' : video_cnt, 'frame_id' : count}
            out['images'].append(image_info)
          
          if terms[0] == prev_frame_id:
            if(terms[2] == "DontCare"):
              cat_id = 0
            if(terms[2] == "Van"):
              cat_id = 1
            if(terms[2] == "Car"):
              cat_id = 2
            if(terms[2] == "Pedestrian"):
              cat_id = 3
            if(terms[3] == "Cyclist"):
              cat_id = 4
            track_id = terms[1]
            box = terms[6:10]
            fbox = []
            fbox.append(float(box[0])) #Top Left x
            fbox.append(float(box[1])) #Top Left y
            fbox.append(float(box[2])-float(box[0])) #width
            fbox.append(float(box[3])-float(box[1])) #height
            ann = {"id" : ann_cnt, 
                   "category_id" : cat_id,
                   "image_id" : image_cnt,
                   "track_id" : track_id,
                   "box" : fbox,
                   "area" : fbox[2]*fbox[3],
                   "iscrowd" : 0          #Change this
            }
            ann_cnt += 1
            out["annotations"].append(ann)
          
          if terms[0] != prev_frame_id:
            prev_frame_id = terms[0]
            image_cnt += 1
            count = 0
            i -= 1

    json.dump(out, open(out_path, 'w'))