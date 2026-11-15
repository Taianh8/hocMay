"""Đối chiếu ID, dự đoán, chỉ số và nguồn gốc của đường học đã lưu."""
import hashlib
import json
import numpy as np
import pandas as pd
from src.pipeline_v3 import official_metadata
from src.utils import ROOT, digest
from scripts.duong_hoc import OUT, SIZES, score


def main():
    labels, ids = official_metadata(); train_ids=set(ids['train']); forbidden=set(ids['val']) | set(ids['test'])
    subsets=json.loads((OUT/'fold_va_mau.json').read_text()); predictions=json.loads((OUT/'du_doan.json').read_text()); provenance=json.loads((OUT/'nguon_goc.json').read_text()); data=pd.read_csv(OUT/'ket_qua_theo_fold.csv'); summary=pd.read_csv(OUT/'tong_hop.csv')
    for name,expected in provenance['sha256'].items():
        if digest(ROOT/name) != expected: raise ValueError(f'Checksum không khớp: {name}')
    h=hashlib.sha256()
    for image_id in sorted(train_ids): h.update(str(image_id).encode()); h.update(digest(ROOT/'data/raw/jpg'/f'image_{image_id:05d}.jpg').encode())
    if h.hexdigest() != provenance['train_images_sha256']: raise ValueError('Nội dung ảnh train đã thay đổi')
    if len(subsets)!=20 or len(predictions)!=60 or len(data)!=60: raise ValueError('Số lượng kết quả không hợp lệ')
    seen=set()
    for item in predictions:
        key=(item['experiment'],item['fold'],item['images_per_class'])
        if key in seen: raise ValueError(f'Kết quả trùng: {key}')
        seen.add(key)
        match=[s for s in subsets if s['fold']==key[1] and s['images_per_class']==key[2]]
        if len(match)!=1: raise ValueError(f'Thiếu tập con: {key}')
        sub=match[0]; fit=np.asarray(sub['train_image_ids']); hold=np.asarray(sub['holdout_image_ids'])
        if len(fit)!=102*key[2] or len(hold)!=204 or len(set(fit))!=len(fit) or len(set(hold))!=len(hold) or set(fit)&set(hold) or not (set(fit)|set(hold)) <= train_ids or (set(fit)|set(hold)) & forbidden:
            raise ValueError(f'Rò rỉ hoặc số lượng sai: {key}')
        if not np.array_equal(np.bincount(labels[fit-1],minlength=102),np.full(102,key[2])) or not np.array_equal(np.bincount(labels[hold-1],minlength=102),np.full(102,2)):
            raise ValueError(f'Tập con không phân tầng: {key}')
        train_pred=np.asarray(item['train_prediction']); hold_pred=np.asarray(item['holdout_prediction'])
        if train_pred.shape != fit.shape or hold_pred.shape != hold.shape or not (((train_pred>=0)&(train_pred<102)).all() and ((hold_pred>=0)&(hold_pred<102)).all()):
            raise ValueError(f'Dự đoán không hợp lệ: {key}')
        train_acc,train_f1=score(labels[fit-1],train_pred); hold_acc,hold_f1=score(labels[hold-1],hold_pred)
        row=data[(data.experiment==key[0])&(data.fold==key[1])&(data.images_per_class==key[2])]
        if len(row)!=1 or not np.allclose([train_acc,train_f1,hold_acc,hold_f1],row[['train_accuracy','train_macro_f1','holdout_accuracy','holdout_macro_f1']].to_numpy()[0],rtol=0,atol=1e-12):
            raise ValueError(f'Chỉ số không khớp dự đoán: {key}')
    expected={(name,fold,n) for name in ('E1','E4','E6') for fold in range(1,6) for n in SIZES}
    if seen != expected: raise ValueError('Thiếu cấu hình kết quả')
    for fold in range(1,6):
        sets=[set(next(s['train_image_ids'] for s in subsets if s['fold']==fold and s['images_per_class']==n)) for n in SIZES]
        if not all(sets[i] <= sets[i+1] for i in range(3)): raise ValueError('Tập con không lồng nhau')
    for row in summary.itertuples():
        group=data[(data.experiment==row.experiment)&(data.train_samples==row.train_samples)]
        if len(group)!=5 or not np.allclose([group.train_macro_f1.mean(),group.holdout_macro_f1.mean(),group.holdout_macro_f1.std(ddof=0)],[row.train_f1,row.holdout_f1,row.holdout_f1_std],rtol=0,atol=1e-12):
            raise ValueError('Bảng tổng hợp không khớp các fold')
    previous=json.loads((ROOT/'results/v3/metrics/cv_folds.json').read_text())
    for row in data[data.train_samples==816].itertuples():
        matched=[r for r in previous if r['experiment']==row.experiment and r['fold']==row.fold and (r['pca_seed'] is None or r['pca_seed']==42)]
        if len(matched)!=1 or abs(matched[0]['macro_f1']-row.holdout_macro_f1)>1e-12: raise ValueError('Kết quả 816 ảnh không khớp V3')
    print('Đã kiểm tra 60 dự đoán, split, điểm, nguồn gốc và đối chiếu V3; không đánh giá test.')

if __name__ == '__main__': main()
