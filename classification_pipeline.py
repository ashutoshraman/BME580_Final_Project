import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import sys


df = pd.read_csv('preprocessed_model_data.csv', delimiter=',')


def pca_svm_pipeline(variance):
    pipeline_obj = make_pipeline(StandardScaler(), PCA(n_components=variance), SVC(kernel='rbf', random_state=10))
    svm_param_grid = [{'svc__C': [0.5, 1, 5, 10, 30, 40],
                       'svc__gamma': ['scale', 'auto', .001, .005, .01],
                       }]
    return pipeline_obj, svm_param_grid

def pca_logistic_pipeline(variance):
    pipeline_obj = make_pipeline(StandardScaler(), PCA(n_components=variance), LogisticRegression(max_iter=1000, random_state=10))
    lr_param_grid = [{'logisticregression__C': [.001, .01, .1, 1],
                      'logisticregression__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                      'logisticregression__penalty': ['l1', 'l2', 'none']}]
    return pipeline_obj, lr_param_grid

def ml_pipeline(pipeline_choice, X, Y, variance=.9):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=True, random_state = 10, stratify=Y)


    if pipeline_choice == 1:
        pipeline_obj, param_grid = pca_svm_pipeline(variance)
    elif pipeline_choice == 0:
        pipeline_obj, param_grid = pca_logistic_pipeline(variance)
    else:
        print('no classifier for this entry')

    grid_search_obj = GridSearchCV(estimator=pipeline_obj,
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=3, verbose=0)

    scores = cross_val_score(grid_search_obj, X_train, Y_train, scoring='accuracy',cv=5, verbose=0)
    
    best_params_model = grid_search_obj.fit(X_train, Y_train) #best params from model will give hyperparameters
    print("Mean Accuracy for pipeline: {:f}".format(np.mean(scores)))
    print("Stdev of Accuracy for pipeline: {:f}".format(np.std(scores)))
    print(best_params_model.best_params_)
    return best_params_model, X_train, X_test, Y_train, Y_test

def AUC_ROC(y_test, y_score, labels):
    def plot_roc_curve(fpr, tpr, label=labels): 
        plt.plot(fpr, tpr, linewidth=4, label=label) 
        
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
    auc = roc_auc_score(y_test, y_score)
    
    plot_roc_curve(fpr, tpr)
    return fpr, tpr, auc

def learning_curve_graph(fitted_pipeline, X_train, Y_train): #possibly include for bias variance trade off showing
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(fitted_pipeline, X_train, Y_train,
                                            train_sizes=np.linspace(.1,1.0,10), cv=4, n_jobs=1, shuffle=False)

    train_mean = np.mean(train_scores,axis=1)
    train_std = np.std(train_scores,axis=1)
    test_mean = np.mean(test_scores,axis=1)
    test_std = np.std(test_scores,axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean+train_std,train_mean-train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean+test_std,test_mean-test_std, alpha=0.15, color='green')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Accuracy of Model Training and Testing (%)')
    plt.title('Learning Curve for Model and Specific Dataset')
    plt.legend()
    plt.show()

def loadingplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = df.iloc[:, -1])
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.title('PCA Biplot for First 2 PCs')
    plt.grid()


scaler = StandardScaler()
pca2 = PCA(n_components=.9)
X_train_scaled = scaler.fit_transform(df.iloc[:, 1:8])
X_train_scaled_reduced = pca2.fit_transform(X_train_scaled)

print(pca2.explained_variance_)
print(pca2.explained_variance_ratio_)
print(pca2.explained_variance_ratio_.cumsum())

loadingplot(X_train_scaled_reduced[:,0:2],np.transpose(pca2.components_[0:2, :]), labels=list(df)[1:8])
plt.show()

pca_one = X_train_scaled_reduced[:, 0].reshape(X_train_scaled_reduced.shape[0])
pca_two = X_train_scaled_reduced[:, 1].reshape(X_train_scaled_reduced.shape[0])


plt.figure()
sns.scatterplot(
                x = pca_one, y= pca_two,
                hue = df.iloc[:, -1],
                style = df.iloc[:, -1],
                palette = sns.color_palette("hls", 2),
                alpha = 0.3
                )
plt.title('PCA One vs Two for VO2 Predictors')
plt.xlabel('PCA One')
plt.ylabel('PCA Two')
plt.show()


#single run...  
best_params, X_train, X_test, Y_train, Y_test = ml_pipeline(1, df.iloc[:, 1:8], df.iloc[:, -1])
print(best_params.best_score_)

# accuracy and evaluation of model
train_score = best_params.score(X_train, Y_train) #why is this not same as best_score_ above??
test_score = best_params.score(X_test, Y_test)
print(train_score, test_score)

#predict a label from test data
Y_pred = best_params.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
# print("Confusion Matrix: \n")
# print(cm)

#confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(best_params, X_test, Y_test,
                                cmap=plt.cm.Blues,
                                normalize=normalize)
    disp.ax_.set_title(title)

    plt.show()

print(classification_report(Y_test, Y_pred))



#roc and auc score

plt.figure()
fpr, tpr, auc = AUC_ROC(Y_test, Y_pred, 1) #change arg 3 to get names for graph legend
print('The AUC for {} is {}'.format(1, auc))

plt.plot([0, 1], [0, 1], 'k--') 
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# call learning curve function producer
# learning_curve_graph(best_params, X_train, Y_train)

# end single run, stop commenting

#this part is to get the overlaid ROC curves
models = ['Logistic Regression', 'SVM']

plt.figure()
for i, n in enumerate(models):
    best_params, X_train, X_test, Y_train, Y_test = ml_pipeline(i, df.iloc[:, 1:8], df.iloc[:, -1])
    Y_pred = best_params.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label=1)
    auc = roc_auc_score(Y_test, Y_pred)    
    plt.plot(fpr, tpr, linewidth=4, label=n)

    print('The AUC for {} is {}'.format(n, auc))
    input('ready for next model?: ')

 #change arg 3 to get names for graph legend

plt.plot([0, 1], [0, 1], 'k--') 
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#plot SVM decision boundary-- figure this out for gridsearch
best_params, X_train, X_test, Y_train, Y_test = ml_pipeline(1, df.iloc[:, 1:8], df.iloc[:, -1], variance=2)
scaler = StandardScaler()
pca2 = PCA(n_components=2)
X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled_reduced = pca2.fit_transform(X_test_scaled)



svm_model = SVC(kernel='rbf', C=best_params.best_params_['svc__C'], gamma=best_params.best_params_['svc__gamma'])
classify = svm_model.fit(X_test_scaled_reduced, Y_test) 

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    print ('initial decision function shape; ', np.shape(Z))
    Z = Z.reshape(xx.shape)
    print ('after reshape: ', np.shape(Z))
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

X0, X1 = X_test_scaled_reduced[:, 0], X_test_scaled_reduced[:, 1] 
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots()#figsize=(12,9))
fig.patch.set_facecolor('white')
cdict1={0:'lime',1:'deeppink'}
labl1={1:'Athletic VO2',0:'Unathletic VO2'}
marker1={0:'*',1:'d'}
alpha1={0:.8, 1:0.5}

Y_tar_list = Y_test.tolist()
labels1= [int(target1) for target1 in Y_tar_list]

for l1 in np.unique(labels1):
    ix1=np.where(labels1==l1)
    ax.scatter(X0[ix1],X1[ix1], c=cdict1[l1],label=labl1[l1],s=70,marker=marker1[l1],alpha=alpha1[l1]) #plot test pts in decision boundary

ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none', 
           edgecolors='navy', label='Support Vectors') #may not be necessary, since this plots support vectors, coming from train set usually

plot_contours(ax, classify, xx, yy,cmap='seismic', alpha=0.4) #this is the boundary built, or model
plt.legend()#fontsize=15)
plt.xlabel("1st Principal Component")#,fontsize=14)
plt.ylabel("2nd Principal Component")#,fontsize=14)
plt.title('SVM Decision Boundary and Predictions for Athleticism')
plt.show()