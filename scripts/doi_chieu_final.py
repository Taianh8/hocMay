"""Đối chiếu output test đã lưu; tuyệt đối không gọi model.predict."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from src.pipeline_v3 import official_metadata
from src.utils import ROOT, digest, save_json
from src.pipeline_v5 import select_best

OUT=ROOT/'results/final_test'


def write_json_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Artifact đã tồn tại nhưng khác kết quả: {path}')
    else:
        save_json(path, value)


def write_csv_once(path, frame):
    if path.exists():
        try:
            pd.testing.assert_frame_equal(pd.read_csv(path), frame.reset_index(drop=True), check_dtype=False, rtol=1e-12, atol=1e-12)
        except AssertionError as exc:
            raise ValueError(f'Artifact đã tồn tại nhưng khác kết quả: {path}') from exc
    else:
        frame.to_csv(path, index=False)

def summary(y,pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,labels=range(102),average='macro',zero_division=0)
    pw,rw,fw,_=precision_recall_fscore_support(y,pred,labels=range(102),average='weighted',zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_precision':float(p),'macro_recall':float(r),'macro_f1':float(f),'weighted_precision':float(pw),'weighted_recall':float(rw),'weighted_f1':float(fw),'samples':len(y)}

def main():
    labels,ids=official_metadata(); expected=np.sort(ids['test']); audit=pd.read_csv(ROOT/'results/v4/duplicate_candidates.csv')
    exact=audit[audit.file_exact]; dup=set()
    for r in exact.itertuples():
        if r.split_a=='test':dup.add(r.image_id_a)
        if r.split_b=='test':dup.add(r.image_id_b)
    if dup!={7309,8067}:raise ValueError(f'Ảnh trùng không khớp kiểm toán: {dup}')
    stored=json.loads((OUT/'metrics.json').read_text()); checked={}
    for exp,key in [('E1','E1_baseline'),('E6','E6_final_model')]:
        data=pd.read_csv(OUT/f'{exp}_test_predictions.csv').sort_values('image_id'); image_ids=data.image_id.to_numpy(); y=data.true_class_id.to_numpy()-1;pred=data.pred_class_id.to_numpy()-1
        if not np.array_equal(image_ids,expected) or not np.array_equal(y,labels[image_ids-1]) or not ((pred>=0)&(pred<102)).all():raise ValueError(f'{exp}: ID/nhãn test không khớp metadata')
        mask=~np.isin(image_ids,list(dup)); computed={'full_test':summary(y,pred),'clean_test_sensitivity':summary(y[mask],pred[mask])}
        for group,metrics in computed.items():
            for name,value in metrics.items():
                if not np.isclose(value,stored[key][group][name],rtol=0,atol=1e-12):raise ValueError(f'{exp}: {group}.{name} không khớp')
        cm=confusion_matrix(y,pred,labels=range(102))
        matrix_file=OUT/f'{exp}_confusion_matrix.npy'
        if matrix_file.exists() and not np.array_equal(cm,np.load(matrix_file)):raise ValueError('Ma trận nhầm lẫn không khớp')
        if exp=='E1':
            p,r,f,s=precision_recall_fscore_support(y,pred,labels=range(102),zero_division=0)
            write_csv_once(OUT/'E1_per_class.csv', pd.DataFrame({'class_id':np.arange(1,103),'precision':p,'recall':r,'f1':f,'support':s}))
            pairs=[{'true_class_id':i+1,'pred_class_id':j+1,'count':int(cm[i,j])} for i in range(102) for j in range(102) if i!=j and cm[i,j]]
            write_csv_once(OUT/'E1_confusion_pairs.csv', pd.DataFrame(pairs).sort_values('count',ascending=False))
        checked[exp]=computed
    comp=json.loads((ROOT/'results/v5/unified_comparison.json').read_text()); winner=select_best(comp['unified_comparison'])
    corrected={'selection_rule':'validation macro F1, tie-break accuracy; all E1–E7 and control variants', 'validation_winner':winner['experiment'],'evaluated_test_model':'E6','winner_not_evaluated_on_test':winner['experiment']!='E6', 'unified_comparison':comp['unified_comparison'], 'source_sha256':digest(ROOT/'results/v5/unified_comparison.json')}
    dest=ROOT/'results/v5/unified_comparison_v2.json'
    write_json_once(dest,corrected)
    write_json_once(OUT/'verification.json',{'checked':checked,'duplicate_ids_from_audit':sorted(dup),'new_model_predictions':False,'validation_winner':winner['experiment'],'test_model':'E6','selection_rule_followed':winner['experiment']=='E6'})
    print('Đã đối chiếu dự đoán đã lưu; không gọi model.predict. Ứng viên validation:',winner['experiment'])

if __name__=='__main__':main()
