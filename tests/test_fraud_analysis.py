import numpy as np
import pandas as pd
import pytest
from fraud_analysis import load_data, metrics, select_threshold

def test_duplicate_transactions_removed_before_any_split(tmp_path):
    data=pd.DataFrame(np.zeros((3,30)),columns=['Time',*[f'V{i}' for i in range(1,29)],'Amount'])
    data['Class']=[0,0,1]
    path=tmp_path/'creditcard.csv'; data.to_csv(path,index=False)
    assert len(load_data(path))==2
    data.loc[0,'Time']=float('nan'); data.to_csv(path,index=False)
    with pytest.raises(ValueError,match='complete'): load_data(path)

def test_validation_threshold_and_confusion_counts():
    y=np.array([0,0,1,1]); p=np.array([.1,.3,.6,.8])
    threshold=select_threshold(y,p)
    assert threshold==.6
    result=metrics(y,p,threshold)
    assert result['false_negative']==result['false_positive']==0
    assert result['true_positive']==result['true_negative']==2
    assert metrics(y,p,.7)['false_negative']==1
