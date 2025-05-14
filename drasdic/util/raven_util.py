import pandas as pd
import numpy as np
from drasdic.util.graph_matching import match_events

def frames_to_st_dict(x, sr):
    # x : Tensor of shape (batch, time) or (time,). Entries are 2 (POS), 1 (UNK), and 0 (NEG).
    # returns a list of dicts {"Begin Time (s)" : [...], "End Time (s)" : [...], "Annotation" : [...]} if batch dim exists, or a single dict
    
    if len(x.size()) == 2:
        outs = []
        for i in range(x.size(0)):
            x_sub = x[i,:]
            outs.append(_frames_to_st_dict_single(x_sub, sr=sr))
        return outs
    else:
        return _frames_to_st_dict_single(x, sr=sr)

def _frames_to_st_dict_single(x, sr):
    d = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}
    
    for label_i in [1,2]:
        
        labels = x.cpu().numpy() == label_i  # POS : 2, UNK : 1, NEG : 0

        starts = np.where((~labels[:-1]) & (labels[1:]))[0] + 1
        if labels[0]:
            starts = np.insert(starts, 0, 0)

        ends = np.where((labels[:-1]) & (~labels[1:]))[0] + 1
        if labels[-1]:
            ends = np.append(ends, len(labels))

        for start, end in zip(starts, ends):
            d["Begin Time (s)"].append(start/sr)
            d["End Time (s)"].append(end/sr)
            d["Annotation"].append("POS" if label_i == 2 else "UNK")
            
    return d

def get_matching_metrics(preds_st_dict, gt_st_dict, unknown_label = "UNK", iou_thresh = 0.5):
    preds_st = pd.DataFrame(preds_st_dict)
    gt_st = pd.DataFrame(gt_st_dict)
    
    # Bipartite graph matching (fast approximate version)
    ref = np.array(gt_st[['Begin Time (s)', 'End Time (s)']]).T
    est = np.array(preds_st[['Begin Time (s)', 'End Time (s)']]).T
    matching = match_events(ref, est, min_iou=iou_thresh, method="fast")
    
    out = {'TP':0, 'FP':0, 'FN' : 0}
    pred_label = np.array(preds_st['Annotation'])
    annot_label = np.array(gt_st['Annotation'])
    
    for p in matching:
        annotation = annot_label[p[0]]
        prediction = pred_label[p[1]]

        if annotation == prediction:
            out['TP'] += 1
        elif (unknown_label is not None) and (annotation == unknown_label):
            out['FP'] -= 1 #adjust FP for unknown labels
    
    n_annot = int((annot_label != unknown_label).sum()) if (len(annot_label)>0) else 0
    n_pred = int((pred_label != unknown_label).sum()) if (len(pred_label)>0) else 0
    out['FP'] = out['FP'] + n_pred - out['TP']
    out['FN'] = n_annot - out['TP']

    return out
    
