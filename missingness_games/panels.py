"""Native-completeness and activity-rarity panels with subject-first minima."""
import numpy as np
def macro_accuracy(y,predicted):
    return float(np.mean([np.mean(predicted[y==label]==label) for label in np.unique(y)]))


def panels(probabilities,data,boundary,rare,represented,heldout):
    native=np.asarray(data.get('mask',np.isfinite(data['x'])),bool)&np.isfinite(data['x'])
    completeness=native.mean((1,2));low=completeness<boundary
    rarity=np.isin(data['y'],rare);records=[]
    for complete_name,complete in [('lower',low),('higher',~low)]:
        for rarity_name,activity in [('all',np.ones(len(low),bool)),('rare',rarity),('common',~rarity)]:
            subject_rows=[]
            for subject in np.unique(data['group']):
                selected=(data['group']==subject)&complete&activity
                if not selected.any():continue
                scores={condition:float(macro_accuracy(data['y'][selected],p[selected].argmax(1))) for condition,p in probabilities.items()}
                subject_rows.append(dict(subject=int(subject),windows=int(selected.sum()),clean=scores['clean'],
                    represented_worst=min(scores[c] for c in represented),heldout_worst=min(scores[c] for c in heldout),
                    low_support_classes=[int(label) for label in np.unique(data['y'][selected]) if ((data['y']==label)&selected).sum()<20],
                    class_support={str(int(label)):int(((data['y']==label)&selected).sum()) for label in np.unique(data['y'][selected])}))
            records.append(dict(completeness=complete_name,rarity=rarity_name,subjects=subject_rows,
                eligible_subjects=len(subject_rows),**{metric:float(np.mean([row[metric] for row in subject_rows])) if subject_rows else None for metric in ('clean','represented_worst','heldout_worst')}))
    return dict(training_native_boundary=float(boundary),rare_classes=list(map(int,rare)),panels=records,
        aggregation='macro class recall within subject/stratum/policy; policy minimum; eligible-subject mean')
