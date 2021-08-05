import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn import metrics
import math

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

""" ---------------
    preprocess data
    ---------------
"""
def get_data_subset(df, sample_fraction, gender, period):
    print 'original data.shape:', df.shape
    
    df = df.drop(['patid', 'pracid'], axis = 1)
    
    if (gender == 0) | (gender == 1):
        df = df[df.gender == gender]
        df = df.drop(['gender'], axis = 1)
    
    df = df.sample(frac=sample_fraction).reset_index(drop=True)
    
    df.loc[:, 'outcome_admission'] = df['outcome_admission_' + period]
    df.loc[:, 'eventdate'] = df['eventdate_' + period]
    
    for col in df.columns.values:
        if 'outcome_admission_' in col:
            df = df.drop([col], axis=1)
        if 'eventdate_' in col:
            df = df.drop([col], axis=1)   
                
    print 'selected data.shape:', df.shape
    
    return df

def get_dummies(db):
    print 'data.shape befor processing:', db.shape
    
    if 'eventdate' in db.columns.values:
        del db['eventdate']
        
    db = db.fillna(-1)
    db = pd.get_dummies(db, columns=['region', 'ethnicity', 'smoking', 'alcohol'])
    
    print 'data.shape after processing:', db.shape
    
    return db

def get_data_in_study_period(db, period):

    db.loc[:, 'outcome_admission'] = db['outcome_admission_' + period]
    db.loc[:, 'eventdate'] = db['eventdate_' + period]
    
    for col in db.columns.values:
        if 'outcome_admission_' in col:
            db = db.drop([col], axis=1)
        if 'eventdate_' in col:
            db = db.drop([col], axis=1)
            
    print 'data.shape', db.shape
    
    return db

""" -------------------------
    fitting linear regression
    -------------------------
"""
def LR_ER(db, k, verbose):    
    train_auc = []
    test_auc = []
    test_results = []
    
    print ('Linear Regression')
    
    #     fitting logistic regression
    for i in range(0,k):
        if verbose == True:
            print ('fold:', i)
            
        split_point = db.shape[0]/k

        # shuffle
        db = db.sample(frac=1).reset_index(drop=True)

        data_test = db.loc[:split_point,:]
        data_train = db.loc[split_point:,:]

        X_train = data_train.drop('outcome_admission', axis=1)
        Y_train = data_train['outcome_admission']

        X_test = data_test.drop('outcome_admission', axis=1)
        Y_test = data_test['outcome_admission']

        # create and fit two linear models with L1 and L2 penalties   
        clf_l1_LR = linear_model.LogisticRegression(penalty='l1')
        clf_l1_LR.fit(X_train, Y_train)
        coef_l1_LR = clf_l1_LR.coef_.ravel()
        sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100

        # accuracy
        Y_pred_train = clf_l1_LR.predict_proba(X_train)
        fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_pred_train[:, 1])
        train_auc.append(metrics.auc(fpr, tpr))
        Y_pred_test = clf_l1_LR.predict_proba(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred_test[:, 1])
        test_auc.append(metrics.auc(fpr, tpr))

        this_fold_results = pd.DataFrame(zip(Y_test, Y_pred_test[:, 0], Y_pred_test[:, 1]), 
                                         columns=['y_true', 'y_pred0', 'y_pred1'])
        test_results.append(this_fold_results)
        
        print ('fold', i, ' train auc', train_auc[i])
        print ('fold', i, ' test auc', test_auc[i])

    print ''
    print '--------------------'
    print '* average train auc', np.mean(train_auc)
    print '* average test auc', np.mean(test_auc)
    print '--------------------'
    print ''
    
    return clf_l1_LR, X_train.columns.values, test_results


""" ---------------------
    fitting random forest
    ---------------------
"""
def RF_ER(db, k, n_estimator, min_samples_split, max_features, verbose):
    train_auc = []
    test_auc = []
    test_results = []
        
    print ('Random Forest')
    
    #     fitting random forrest
    for i in range(0,k):
        if verbose == True:
            print ('fold:', i)
            
        split_point = db.shape[0]/k

        # shuffle
        db = db.sample(frac=1).reset_index(drop=True)

        data_test = db.loc[:split_point,:]
        data_train = db.loc[split_point:,:]

        X_train = data_train.drop('outcome_admission', axis=1)
        Y_train = data_train['outcome_admission']

        X_test = data_test.drop('outcome_admission', axis=1)
        Y_test = data_test['outcome_admission']


        clf = RandomForestClassifier(n_estimators = n_estimator, 
                                     min_samples_split = min_samples_split,
                                     max_features = max_features)

        clf = clf.fit(X_train, Y_train)

        # accuracy
        Y_pred_train = clf.predict_proba(X_train)
        fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_pred_train[:, 1])
        train_auc.append(metrics.auc(fpr, tpr))

        Y_pred_test = clf.predict_proba(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred_test[:, 1])
        test_auc.append(metrics.auc(fpr, tpr))

        this_fold_results = pd.DataFrame(zip(Y_test, Y_pred_test[:, 0], Y_pred_test[:, 1]), 
                                         columns=['y_true', 'y_pred0', 'y_pred1'])
        test_results.append(this_fold_results)
        
        print ('fold', i, ' train auc', train_auc[i])
        print ('fold', i, ' test auc', test_auc[i])

    print ''
    print '--------------------'
    print '* average train auc', np.mean(train_auc)
    print '* average test auc', np.mean(test_auc)
    print '--------------------'
    print ''
    
    return clf, X_train.columns.values, test_results


""" -------------------------
    fitting gradient boosting
    -------------------------
"""
def GB_ER(db, k, n_estimator, min_samples_split=0.001, max_features=0.5, max_depth=3, verbose=True):
    train_auc = []
    test_auc = []
    test_results = []

    single_run = False
    
    if k == 1:
        single_run = True
        k = 3
        
    print ('Gradient Boosting')

    #     fitting gradient boosting
    for i in range(0,k):
        if verbose == True:
            print ('fold:', i)
            
        split_point = db.shape[0]/k

        # shuffle
        db = db.sample(frac=1).reset_index(drop=True)

        data_test = db.loc[:split_point,:]
        data_train = db.loc[split_point:,:]

        X_train = data_train.drop('outcome_admission', axis=1)
        Y_train = data_train['outcome_admission']

        X_test = data_test.drop('outcome_admission', axis=1)
        Y_test = data_test['outcome_admission']

        clf = GradientBoostingClassifier(n_estimators = n_estimator
                                         , max_features= max_features
                                         , min_samples_split = min_samples_split
                                         , max_depth = max_depth
                                         , verbose = verbose)
        
        clf = clf.fit(X_train, Y_train)

        # accuracy
        Y_pred_train = clf.predict_proba(X_train)
        fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_pred_train[:, 1])
        train_auc.append(metrics.auc(fpr, tpr))

        Y_pred_test = clf.predict_proba(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred_test[:, 1])
        test_auc.append(metrics.auc(fpr, tpr))

        this_fold_results = pd.DataFrame(zip(Y_test, Y_pred_test[:, 0], Y_pred_test[:, 1]), 
                                         columns=['y_true', 'y_pred0', 'y_pred1'])
        test_results.append(this_fold_results)
        
        print ('fold', i, ' train auc', train_auc[i])
        print ('fold', i, ' test auc', test_auc[i])
    
        if single_run == True:
            break
    
    print ''
    print '--------------------'
    print '* average train auc', np.mean(train_auc)
    print '* average test auc', np.mean(test_auc)
    print '--------------------'
    print ''
    
    return clf, X_train.columns.values, test_results


""" ------------------------
    print feature importance
    ------------------------
"""
def print_feature_importance(clf, labels, n):
    if isinstance(clf, linear_model.LogisticRegression):
        return
    else:
        importances = clf.feature_importances_
        
    print importances.shape
    
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print ("")
    print("Feature ranking:")

    i = 1
    for f in indices:
        print("%d. %s (%f)" % (i, labels[f], importances[f]))
        i += 1
        if i > n:
            break
    return


""" ----------------------
    Save results to csv
    ----------------------
"""
def save_results(results, k, filename):
    df_out = pd.DataFrame(columns = ['y_true', 'y_pred', 'fold'])
    for i in range(k):
        x = results[i]
        x['fold'] = i
        df_out = df_out.append(x)
    print 'result shape:', df_out.shape, 'saving... (this will take quite some time)'
    df_out.to_csv(filename, index = False)
    print 'saved!'
    
    
""" ----------------------
    Save/Load classifier
    ----------------------
"""    
from sklearn.externals import joblib

def save_CLF(clf, filename):
    output = joblib.dump(clf, filename, compress=9)
    print output
    return

def load_CLF(filename):
    return joblib.load(filename)
    
    
""" ----------------------
    Predict
    ----------------------
"""    
def predict(clf, data):
    X = data.drop('outcome_admission', axis=1)
    y = data['outcome_admission']
    
    y_pred = clf.predict_proba(X)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred[:, 1])
        
    print metrics.auc(fpr, tpr)
    
    results = pd.DataFrame(zip(y, y_pred[:, 0], y_pred[:, 1]), 
                           columns=['y_true', 'y_pred0', 'y_pred1'])
    
    return results
    
""" ----------------------
    Plots
    ----------------------
"""

def plot_ALL(k, Y_results, label):
    plot_ROC(k, Y_results, label)
    plot_Calibration_Plot(k, Y_results, label)
    plot_Calibration_Centiles(k, Y_results, label)
    plot_Precision_Recall_Curve(k, Y_results, label)
    return

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn import metrics

def plot_ROC(k, Y_results, label):
    colors = ['red', 'navy', 'green', 'darkorange', 'magneta', 'cyan']
    avg_auc = 0
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    
    for i in range(k):
        y_true_fold = Y_results[i].y_true
        y_pred_fold = Y_results[i].y_pred1
        fpr, tpr, thresholds = metrics.roc_curve(y_true_fold, y_pred_fold)
        roc_auc = metrics.auc(fpr, tpr)
        avg_auc += roc_auc
    
        lw = 1
        #plt.plot(fpr, tpr, color=colors[i],
        #     lw=lw, label='fold %(fold)d (area = %(auc)0.2f)'% {'fold':i , 'auc':roc_auc})
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std


    plt.plot(base_fpr, mean_tprs, colors[i], lw = lw, label = label)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.3)
    
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    avg_auc = avg_auc / k
    plt.title('Receiver operating characteristic (Average AUC=%0.3f)' % avg_auc)
    plt.legend(loc="lower right")
    
    plt.show()
    return


def plot_Calibration_Plot(k, Y_results, label):
    colors = ['red', 'navy', 'green', 'darkorange', 'magneta', 'cyan']
    avg_auc = 0
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    
    for i in range(k):
        y_true_fold = Y_results[i].y_true
        y_pred_fold = Y_results[i].y_pred1
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_fold, y_pred_fold, n_bins=20)
        #plt.plot(mean_predicted_value, fraction_of_positives, "s-", color= colors[i],
        #    label='fold %d '% (i))
        tpr = interp(base_fpr, mean_predicted_value, fraction_of_positives)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std        
    
    plt.plot(base_fpr, mean_tprs, colors[i], label = label)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.1)
    
    lw = 1    
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--', label="Perfectly calibrated")    
    plt.xlim([0.0, 1.0])
    plt.xlabel('Produced probability')
    plt.ylabel('Fraction of positives')
    plt.ylim([-0.05, 1.05])
    plt.legend(loc='upper left')
    plt.title('Calibration plot  (reliability curve)')
    
    plt.show()
    return


def plot_Calibration_Centiles(k, Y_results, label):
    colors = ['red', 'navy', 'green', 'darkorange', 'magneta', 'cyan']
    cnt = 10
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    
    for i in range(k):
        y_true_fold = Y_results[i].y_true
        y_pred_fold = Y_results[i].y_pred1
        ranked_prob_index = np.argsort(y_pred_fold)
        ranked_prob = []
        x_labels = []
        y_label = []
        y_prime_label = []
        
        centile_size = int(ranked_prob_index.shape[0]/cnt)
        for j in range(cnt):
            l = int(j * centile_size)
            if j == cnt-1:
                h = ranked_prob_index.shape[0]
            else:
                h = int((j+1) * centile_size)

            index = ranked_prob_index[l:h]

            x_labels.append(j * 1.0 / cnt)
            y_value = (y_true_fold[index]==1).sum() * 1.0 / y_true_fold[index].shape[0]
            y_label.append(y_value)

        #plt.plot(x_labels, y_label, color=colors[i], label='%s fold %d' % (label, i))
        tpr = interp(base_fpr, x_labels, y_label)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std        
    
    plt.plot(base_fpr, mean_tprs, colors[i], label = label)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.3)

    lw=1
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Decile of Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot, based on centiles')
    plt.legend(loc="upper left")
    
    plt.show()
    return

def plot_Precision_Recall_Curve(k, Y_results, label):
    colors = ['red', 'navy', 'green', 'darkorange', 'magneta', 'cyan']
    
    for i in range(k):
        y_true_fold = Y_results[i].y_true
        y_pred_fold = Y_results[i].y_pred1
        precision, recall, _ = precision_recall_curve(y_true_fold, y_pred_fold)
        average_precision = metrics.average_precision_score(y_true_fold, y_pred_fold)
        plt.step(recall, precision, color=colors[i], alpha=1.0, where='post', 
                 label='%s fold %d (area = %0.2f)'% (label, i , average_precision))

    lw = 2    
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([-0.05, 1.05])
    plt.legend(loc='upper right')
    plt.title('2-class Precision-Recall curve')
    
    plt.show()
    return

def plot_ROC_Multiple_CLF(k, Y_results_list, labels):
    colors = ['red', 'navy', 'green', 'darkorange', 'magneta', 'cyan']
    
    for c in range(len(Y_results_list)):
        Y_results = Y_results_list[c]
        avg_auc = 0
        tprs = []
        base_fpr = np.linspace(0, 1, 101)

        for i in range(k):
            y_true_fold = Y_results[i].y_true
            y_pred_fold = Y_results[i].y_pred1
            fpr, tpr, thresholds = metrics.roc_curve(y_true_fold, y_pred_fold)
            roc_auc = metrics.auc(fpr, tpr)
            avg_auc += roc_auc

            lw = 1
            #plt.plot(fpr, tpr, color=colors[i],
            #     lw=lw, label='fold %(fold)d (area = %(auc)0.2f)'% {'fold':i , 'auc':roc_auc})
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        avg_auc = avg_auc / k
        plt.plot(base_fpr, mean_tprs, colors[c], lw = lw, label= '%s (Average AUC=%0.3f)' % (labels[c], avg_auc))
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[c], alpha=0.3)

    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    plt.show()
    return


def plot_Calibration_Plot_multiple_CLF(k, Y_results_list, labels):
    colors = ['red', 'navy', 'green', 'darkorange', 'magneta', 'cyan']
    
    for c in range(len(Y_results_list)):
        Y_results = Y_results_list[c]
        avg_auc = 0
        tprs = []
        base_fpr = np.linspace(0, 1, 101)

        for i in range(k):
            y_true_fold = Y_results[i].y_true
            y_pred_fold = Y_results[i].y_pred1
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true_fold, y_pred_fold, n_bins=20)
            #plt.plot(mean_predicted_value, fraction_of_positives, "s-", color= colors[i],
            #    label='fold %d '% (i))
            tpr = interp(base_fpr, mean_predicted_value, fraction_of_positives)
            tpr[0] = 0.0
            tprs.append(tpr)

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std        

        plt.plot(base_fpr, mean_tprs, colors[c], label= '%s' % labels[c])
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[c], alpha=0.1)
    
    lw = 1    
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--', label="Perfectly calibrated")    
    plt.xlim([0.0, 1.0])
    plt.xlabel('Produced probability')
    plt.ylabel('Fraction of positives')
    plt.ylim([-0.05, 1.05])
    plt.legend(loc='upper left')
    plt.title('Calibration plot  (reliability curve)')
    
    plt.show()
    return


def plot_Calibration_Centiles_multiple_CLF(k, Y_results_list, labels):
    colors = ['red', 'navy', 'green', 'darkorange', 'magneta', 'cyan']
    cnt = 10
    
    for c in range(len(Y_results_list)):
        Y_results = Y_results_list[c]
        avg_auc = 0
        tprs = []
        base_fpr = np.linspace(0, 1, 101)

        for i in range(k):
            y_true_fold = Y_results[0].y_true
            y_pred_fold = Y_results[0].y_pred1
            ranked_prob_index = np.argsort(y_pred_fold)
            ranked_prob = []
            x_labels = []
            y_label = []
            y_prime_label = []

            centile_size = int(ranked_prob_index.shape[0]/cnt)
            for j in range(cnt):
                l = int(j * centile_size)
                if j == cnt-1:
                    h = ranked_prob_index.shape[0]
                else:
                    h = int((j+1) * centile_size)

                index = ranked_prob_index[l:h]

                x_labels.append(j * 1.0 / cnt)
                y_value = (y_true_fold[index]==1).sum() * 1.0 / y_true_fold[index].shape[0]
                y_label.append(y_value)

            tpr = interp(base_fpr, x_labels, y_label)
            tpr[0] = 0.0
            tprs.append(tpr)

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std        
        lw = 1
        plt.plot(base_fpr, mean_tprs, colors[c], lw = lw, label= '%s' % labels[c])
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[c], alpha=0.1)
    
    lw=2
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Decile of Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot, based on centiles')
    plt.legend(loc="upper left")
    
    plt.show()
    return

def plot_Precision_Recall_multiple_CLF(k, Y_results_list, labels):
    colors = ['red', 'navy', 'green', 'darkorange', 'magneta', 'cyan']
    
    for i in range(k):
        y_true_fold = Y_results_list[i].y_true
        y_pred_fold = Y_results_list[i].y_pred1
        precision, recall, _ = precision_recall_curve(y_true_fold, y_pred_fold)
        average_precision = metrics.average_precision_score(y_true_fold, y_pred_fold)
        plt.step(recall, precision, color=colors[i], alpha=1.0, where='post', 
                 label='%s (area = %0.2f)'% (labels[i], average_precision))

    lw = 2    
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([-0.05, 1.05])
    plt.legend(loc='upper right')
    plt.title('2-class Precision-Recall curve')
    
    plt.show()
    return

def plot_ROC_Over_Period(Y_results_list, labels):
    colors = ['red', 'navy', 'green', 'darkorange', 'blue', 'gold', 'indigo', 'sienna', 'cyan']
    output_auc = []
    
    for c in range(len(Y_results_list)):
        Y_results = Y_results_list[c]        
        y_true_fold = Y_results[0].y_true
        y_pred_fold = Y_results[0].y_pred1
        fpr, tpr, thresholds = metrics.roc_curve(y_true_fold, y_pred_fold)
        roc_auc = metrics.auc(fpr, tpr)
        output_auc.append(roc_auc)
        lw = 2
        label = '%s (auc = %0.3f)'% (str(labels[c]), roc_auc)
        plt.plot(fpr, tpr, color=colors[c], lw=lw, label=label)

    lw = 1
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.show()
    
    return output_auc