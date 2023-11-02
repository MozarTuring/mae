import argparse
import json
import os
import sys
import traceback


import numpy as np
from flask import Flask, request, jsonify
import torch
import torch.backends.cudnn as cudnn

import models_vit
from util.datasets import build_dataset
import util.misc as misc

parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
#parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.set_defaults(pin_mem=True)
parser.add_argument('--input_size', default=224, type=int,
                    help='images input size')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')
parser.add_argument('--nb_classes', default=1000, type=int,
                    help='number of the classification types')
parser.add_argument('--global_pool', action='store_true')
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')
parser.set_defaults(global_pool=True)
args = parser.parse_args()

args.resume = "/home/maojingwei/project/resources/pretrained_models/mae_finetuned_vit_base.pth"
args.batch_size = 16
#misc.init_distributed_mode(args)
#cudnn.benchmark = True
device = torch.device(args.device)

model = models_vit.__dict__["vit_base_patch16"](
    num_classes=args.nb_classes,
    drop_path_rate=args.drop_path,
    global_pool=args.global_pool,
)
model.to(device)
checkpoint = torch.load(args.resume, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()



#app=Flask(__name__)
#@app.route('/mae', methods=["POST"])
def mae_interface(inp_data_path):
    try:
#        print("start")
#        inp_param = request.get_data(as_text=True)
#        inp_param = json.loads(inp_param)
#        print("get {}".format(inp_param))
        args.data_path = inp_data_path
        dataset_val = build_dataset(is_train=False, args=args)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        model.eval()
        with torch.no_grad(): # out of memory if without this line
            tmp_ls = list()
            for batch in data_loader_val:
                images = batch[0]
                target = batch[-1]
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                with torch.cuda.amp.autocast():
                    print(images.shape)
                    feat = model.forward_features(images)
                    tmp_ls.append(feat)

        samples = dataset_val.samples
        feats_torch = torch.cat(tmp_ls,dim=0)
        print(feats_torch.shape)
        feats_numpy = feats_torch.cpu().numpy()
        for ind, ele in enumerate(samples):
            tmp_path = os.path.join(inp_data_path, "val_features", ele[0].split("/")[-1].replace("jpeg","npy"))
#            if not os.path.exists(tmp_path):
            print(tmp_path)
            with open(tmp_path, "wb") as wf:
                np.save(wf, feats_numpy[ind]/np.linalg.norm(feats_numpy[ind]))
        print(len(samples))
#        with open(os.path.join(inp_data_path,"val_features.npy"), "wb") as wf:
#            np.save(wf, feats_numpy)
#        with open(os.path.join(inp_data_path,"val_sample_path.txt"), "w", encoding="utf8") as wf:
#            wf.write("\n".join([ele[0] for ele in samples]))
#        return jsonify({"stats": 200})
    except:
        print(traceback.format_exc())
#        return jsonify({"stats":9999})



def use_feat(query_feat_dir=None, standard_feat_dir=None):
    name_ls = list()
    tmp_feat_ls = list()
    for ele in os.listdir(standard_feat_dir):
        name_ls.append(ele.replace(".npy",""))
        tmp_feat = np.load(os.path.join(standard_feat_dir, ele))
        tmp_feat_ls.append(tmp_feat[np.newaxis,:])
    standard_feat = np.concatenate(tmp_feat_ls)

    tmp_ls = ["20230405_133240_李明晓", "20230405_133240_陌生人"]
    for pic in tmp_ls:
        query_feat = np.load(os.path.join(query_feat_dir, pic+".npy"))
        score = np.dot(standard_feat, query_feat)
        max_index = np.argmax(score)
        max_score = score[max_index]
        print(max_score)
        if max_score > 0.25:
            print(name_ls[max_index])
        else:
            print('陌生人')



if __name__=='__main__':
    standard_dir = "/home/maojingwei/project/sribd_attendance/face_database/faces_aligned/"
    query_dir = "/home/maojingwei/project/resources/sribd_attendance_data/body_faces_aligned/"
    for tmp_path in [standard_dir, query_dir]:
        mae_interface(tmp_path)
    use_feat(query_feat_dir=os.path.join(query_dir,"val_features"), standard_feat_dir=os.path.join(standard_dir,"val_features"))
#    app.debug=True 
#    app.run(host='0.0.0.0',port=52401)


