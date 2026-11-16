"""Tạo bảng lỗi từ dự đoán validation đã lưu, không đánh giá model."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from src.flower_catalog import load_catalog
from src.utils import ROOT


def main():
    out=ROOT/'results/v5'; out.mkdir(parents=True,exist_ok=True)
    pred=pd.read_csv(ROOT/'results/v2/metrics/E6_validation_predictions.csv')
    y=pred.class_id.to_numpy()-1; guess=pred.predicted_class_id.to_numpy()-1
    precision,recall,f1,support=precision_recall_fscore_support(y,guess,labels=range(102),zero_division=0)
    saved=pd.read_csv(ROOT/'results/v3/metrics/E6_per_class.csv').sort_values('class_id')
    if not np.allclose(saved.f1,f1,rtol=0,atol=1e-12) or not np.array_equal(saved.support,support):raise ValueError('Chỉ số validation không khớp dự đoán')
    catalog=load_catalog(); frame=pd.DataFrame({'class_id':np.arange(1,103),'name_en':[catalog[i]['name_en'] for i in range(1,103)],'precision':precision,'recall':recall,'f1':f1,'support':support})
    frame.sort_values(['f1','class_id'],ascending=[False,True]).head(10).to_csv(out/'validation_top10_f1.csv',index=False)
    frame.sort_values(['f1','class_id'],ascending=[True,True]).head(10).to_csv(out/'validation_bottom10_f1.csv',index=False)
    wrong=pred[pred.class_id!=pred.predicted_class_id]
    pairs=wrong.groupby(['class_id','predicted_class_id']).size().reset_index(name='count').sort_values(['count','class_id','predicted_class_id'],ascending=[False,True,True]).head(10)
    pairs['true_name_en']=pairs.class_id.map(lambda i:catalog[i]['name_en']);pairs['predicted_name_en']=pairs.predicted_class_id.map(lambda i:catalog[i]['name_en']); pairs['support']=pairs.class_id.map(dict(zip(frame.class_id,frame.support)))
    pairs.to_csv(out/'validation_top10_confusions.csv',index=False)
    print('Đã tạo bảng top 10 validation từ dự đoán đã lưu.')

if __name__=='__main__': main()
