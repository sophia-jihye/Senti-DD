from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def create_confusion_matrix(labels, preds, out_filepath):
    confmat = pd.crosstab(pd.Series(labels, name="true"), pd.Series(preds, name="prediction"))
    confmat['n_true'] = confmat.sum(axis=1)
    confmat.loc['n_pred'] = confmat.sum(axis=0)
    confmat.to_csv(out_filepath)
    print('Created {}'.format(out_filepath))
    return confmat

def create_classification_report(labels, preds, accuracy, out_filepath):
    report = pd.DataFrame(classification_report(labels, preds, output_dict=True)).transpose()
    report.loc['accuracy'] = accuracy
    report.to_csv(out_filepath)
    print('Created {}'.format(out_filepath))
    return report

def average_classification_report(report_filepaths, out_filepath):
    for i, report_filepath in enumerate(report_filepaths):
        report = pd.read_csv(report_filepath, header=0, index_col=0)
        if i == 0: report_avg = report
        else: report_avg = report_avg.add(report)
    report_avg = report_avg / len(report_filepaths)
    report_avg.to_csv(out_filepath)
    print('Created {}'.format(out_filepath))
    return report_avg

def sum_confusion_matrix(conf_filepaths, out_filepath):
    for i, conf_filepath in enumerate(conf_filepaths):
        conf = pd.read_csv(conf_filepath, header=0, index_col=0)
        if i == 0: conf_sum = conf
        else: conf_sum = conf_sum.add(conf)
    conf_sum.to_csv(out_filepath)
    print('Created {}'.format(out_filepath))
    return conf_sum