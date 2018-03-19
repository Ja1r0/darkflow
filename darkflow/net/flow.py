import os
import time
import numpy as np
import tensorflow as tf
import pickle
from multiprocessing.pool import ThreadPool
import xml.etree.ElementTree as ET
import cv2
from collections import namedtuple

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

DiffthreshInfo=namedtuple('DiffthreshInfo',['threshold','outInfo'])
BoxesInfo=namedtuple('BoxesInfo',['img_name','boxesInfo'])
Groundtruth=namedtuple('Groundtruth',['label','gt'])
Compute=namedtuple('Compute',[])
def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)
def build_test(self):
    print('testing yang2')


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)
        # my code begin
        if self.FLAGS.iftest:
            test_path=os.path.join(self.FLAGS.test,'imgs')
            all_imgs=os.listdir(test_path)
            all_imgs=[i for i in all_imgs if self.framework.is_inp(i)]
            h,w,_=cv2.imread(os.path.join(self.FLAGS.test,'imgs/'+all_imgs[0])).shape
            if not all_imgs:
                msg='Failed to find any test images in {} .'
                exit('Error: {}'.format(msg.format(test_path)))
            inp_num=len(all_imgs)
            inp_feed=pool.map(lambda inp: (
                np.expand_dims(self.framework.preprocess(
                    os.path.join(test_path,inp)
                ),0)
            ),all_imgs)
            # feed to the net
            feed_dict_t={self.inp : np.concatenate(inp_feed,0)}
            #self.say('Computing mAP for current model ...')
            out=self.sess.run(self.out,feed_dict_t)
            thresholds=np.linspace(0.01,1,100,endpoint=False)
            diffthreshInfo=list()
            for threshold in thresholds:
                outInfo=list()
                for i in range(out.shape[0]):
                    single_out=out[i]
                    boxesInfo=list()
                    boxes=self.framework.findboxes(single_out)
                    for box in boxes:
                        tmpBox=self.framework.process_box(box,h,w,threshold)
                        if tmpBox is None:
                            continue
                        boxesInfo.append({
                            'label':tmpBox[4],
                            'confidence':tmpBox[6],
                            'topleft':{
                                'x':tmpBox[0],
                                'y':tmpBox[2]},
                            'bottomright':{
                                'x':tmpBox[1],
                                'y':tmpBox[3]}
                        })
                    img_name=all_imgs[i]
                    outInfo.append(BoxesInfo(img_name,boxesInfo))
                diffthreshInfo.append(DiffthreshInfo(threshold,outInfo))

            labels=self.meta['labels']
            count=dict()
            for label_name in labels:
                count[label_name]=dict()
                count[label_name]['TP']=0
                count[label_name]['FP'] = 0
                count[label_name]['FN'] = 0
            pre_for_thresh = list()
            rec_for_thresh = list()
            for info_for_thresh in diffthreshInfo:
                threshold=info_for_thresh.threshold
                pre_for_label = list()
                rec_for_label = list()
                for info_for_img in info_for_thresh.outInfo:
                    img_name=info_for_img.img_name
                    boxes=info_for_img.boxesInfo
                    labels_g,gts=get_gt(img_name,os.path.join(self.FLAGS.test,'xmls'))
                    if len(boxes)==0:
                        for label in labels_g:
                            count[label]['FN']+=1
                    else:
                        labels_p=list()
                        for box in boxes:
                            labels_p.append(box['label'])
                        for i in range(len(labels_g)):
                            label=labels_g[i]
                            if labels_p.count(label)==0:
                                count[label]['FN']+=1
                            elif labels_p.count(label)>1:
                                count[label]['FP']+=1
                            elif labels_p.count(label)==1:
                                box_idx=labels_p.index(label)
                                box=boxes[box_idx]
                                pxmin = box['topleft']['x']
                                pymin = box['topleft']['y']
                                pxmax = box['bottomright']['x']
                                pymax = box['bottomright']['y']
                                pred = (pxmin, pymin, pxmax, pymax)
                                gt=gts[i]
                                iou=compute_iou(gt,pred)
                                if iou>0.5: count[label]['TP']+=1
                                else: count[label]['FN']+=1;count[label]['FP']+=1

                for label_name in labels:
                    if count[label_name]['TP']==0 and count[label_name]['FP']==0:
                        pre=1
                        rec=0
                    elif count[label_name]['TP']==0 and count[label_name]['FN']==0:
                        rec=1
                        pre=0
                    else:
                        pre=count[label_name]['TP']/(count[label_name]['TP']+count[label_name]['FP'])
                        rec=count[label_name]['TP']/(count[label_name]['TP']+count[label_name]['FN'])
                    pre_for_label.append(pre)
                    rec_for_label.append(rec)
            pre_for_thresh.append(pre_for_label)
            rec_for_thresh.append(rec_for_label)
            aps=list()
            for i in range(len(labels)):
                rec=[r[i] for r in rec_for_thresh]
                pre=[p[i] for p in pre_for_thresh]
                ap=voc_ap(rec,pre)
                aps.append(ap)
            mAP=float(sum(aps))/len(aps)
            self.say('mAP= {}'.format(mAP))







    if ckpt: _save_ckpt(self, *args)

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_iou(gt,pred):
    xA=max(gt[0],pred[0])
    yA=max(gt[1],pred[1])
    xB=min(gt[2],pred[2])
    yB=min(gt[3],pred[3])
    inter_area=(xB-xA+1)*(yB-yA+1)
    box1_area=(gt[2]-gt[0]+1)*(gt[3]-gt[1]+1)
    box2_area=(pred[2]-pred[0]+1)*(pred[3]-pred[1]+1)
    iou=inter_area/float(box1_area+box2_area-inter_area)
    return iou

def get_gt(img_name,xml_dir):
    xml_name=os.path.join(xml_dir,img_name.rstrip('.jpg')+'.xml')
    in_file=open(xml_name)
    tree=ET.parse(xml_name)
    root=tree.getroot()
    labels=list()
    gts=list()
    for obj in root.iter('object'):
        label=obj.find('name').text
        labels.append(label)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        gt = (xmin, ymin, xmax, ymax)
        gts.append(gt)
    return labels,gts

'''
def metirc(self):
    anndir='path to testset/xmls'
    imgdir='path to testset/imgs'
    labels=self.FLAGS.labels
    f=open(labels)
    labels=[s.rstrip('\n') for s in f.readlines()]
    os.chdir(anndir)
    anns=os.listdir(anndir)
    for ann in anns:
        in_file=open(ann)
        tree=ET.parse(ann)
        root=tree.getroot()
        img_name=root.find('filename').text
        img=cv2.imread(os.path.join(imgdir,img_name))
        
        boxes=network_out_boxes(self,img)
        thresholds=np.linspace(0.01,1,num=100,endpoint=False)
        for threshold in thresholds:

        for obj in root.iter('object'):
            obj_name=obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            gt = (xmin, ymin, xmax, ymax)
'''


def network_out_boxes(self,anndir='/home/yang/src/darkflow_/test_m1w/xmls',
                      imgdir='/home/yang/src/darkflow_/test_m1w/imgs'):
    anns=os.listdir(anndir)
    imgs=[]
    for ann in anns:
        in_file=open(ann)
        tree=ET.parse(ann)
        root=tree.getroot()
        img_name=root.find('filename').text
        img=cv2.imread(os.path.join(imgdir,img_name))
        #h,w,_=img.shape
        im=self.framework.resize_input(img)
        imgs.append(im)
        in_file.close()
    img_input=np.array(imgs)

    assert isinstance(img_input, np.ndarray), \
        'Image is not a np.ndarray'


    feed_dict = {self.inp: img_input}
    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)

    return boxes


'''
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo
'''

def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)

    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

import math

def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
