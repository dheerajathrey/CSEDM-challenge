import pandas as pd
import numpy as np
import pandas
import csv
import ast
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score , accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
import pickle

input_feat =[0,1,2,3,4,5,6,7,8]
# input_feat = [0,1,2,3,4]
# input_feat = [5,6,7,8]
output_feat = [9]
no_estimators=65

def is_valid_python_file(contents):
    try:
        ast.parse(contents)
        return True
    except SyntaxError:
        return False



def mean(a):
    return sum(a)/len(a)

def basic_model(df):
    train_x =  df.iloc[:,input_feat]
    train_y = df.iloc[:,output_feat]
    train_x =train_x.values
    train_y  =train_y.values
    # clf = MLPClassifier(solver='lbfgs' , alpha=1e-5,hidden_layer_sizes=(100,50,2), random_state=1).fit(train_x,train_y)
    # clf=svm.SVC(kernel='rbf').fit(train_x,train_y)
    clf=DecisionTreeClassifier().fit(train_x,train_y)
    # clf=LogisticRegression(solver='lbfgs')
    model = BaggingClassifier(base_estimator=clf, n_estimators=no_estimators, random_state=7)
    # model = AdaBoostClassifier(base_estimator=clf, n_estimators=no_estimators, learning_rate=5)

    model=model.fit(train_x,train_y)
    return model



df1=pd.read_csv("MainTable.csv")
df2=pd.read_csv("CodeState.csv")
df_merged_code=pd.merge(df1,df2,on="CodeStateID")
df_merged_code=df_merged_code.rename(columns={"Order" : "StartOrder"})


def add_features_basic(df_train):

    df_train = df_train.sort_values(by=['SubjectID'])

    prev_student = None

    p_prior_correct = []
    p_prior_completed = []
    prior_attempts = []

    for index, rows in df_train.iterrows():
        curr_student = rows['SubjectID']

        if(prev_student != curr_student):
            attempts = 0
            first_correct_attempts = 0
            completed_attempts = 0
            
        
        prev_student = curr_student   

        if(attempts > 0):
            p_prior_correct.append(first_correct_attempts/attempts)
            p_prior_completed.append(completed_attempts/attempts)
            prior_attempts.append(attempts)
        else:
            p_prior_correct.append(1/2.0)
            p_prior_completed.append(1/2.0)
            prior_attempts.append(0)
            
        if(rows['FirstCorrect']==True):
            first_correct_attempts+=1

        if(rows['EverCorrect']==True):
            completed_attempts+=1
            
        attempts+=1    
            
    df_train['p_prior_correct'] = p_prior_correct
    df_train['p_prior_completed'] = p_prior_completed
    df_train['prior_attempts'] = prior_attempts

    is_syntax_error = []
    has_fname_error=[]

    for index, rows in df_train.iterrows():
        fname=rows["ProblemID"]

        if(df_train[index:index+1]['Code'].isna().sum()==1):
            is_syntax_error.append(True)
            continue
        
        x = is_valid_python_file(rows['Code'])
        if(x == False):
            is_syntax_error.append(True)
        else:
            is_syntax_error.append(False)
            
    df_train['is_syntax_error'] = is_syntax_error   

    is_semantic_error = []

    for index, rows in df_train.iterrows():
        if(rows['is_syntax_error'] == True):
            is_semantic_error.append('NA')
        elif(rows['is_syntax_error'] == False and rows['Correct'] == False):
            is_semantic_error.append(True)
        else:
            is_semantic_error.append(False)

    df_train['is_semantic_error'] = is_semantic_error

    df_train=df_train.sort_values(["SubjectID"])

    prev_student = None

    p_syntax_errors = []
    p_semantic_errors = []

    for index, rows in df_train.iterrows():
        curr_student = rows['SubjectID']


        if(prev_student != curr_student):
            num_syntax_errors = 0
            num_semantic_errors = 0
            total_attempts = 0

        prev_student = curr_student   

        if(total_attempts == 0):
            p_syntax_errors.append(1.0/3)
            p_semantic_errors.append(1.0/3)

            if(rows['is_syntax_error'] == True):
                num_syntax_errors = num_syntax_errors + 1

            if(rows['is_semantic_error'] == True):
                num_semantic_errors=num_semantic_errors + 1
                
            total_attempts+=1    
        else:
            p_semantic_errors.append(num_semantic_errors/total_attempts)
            p_syntax_errors.append(num_syntax_errors/total_attempts)

            if(rows['is_syntax_error'] == True):
                num_syntax_errors = num_syntax_errors + 1

            if(rows['is_semantic_error'] == True):
                num_semantic_errors=num_semantic_errors + 1
            
            total_attempts+=1


    df_train['pSubjectSyntaxErrors'] = p_syntax_errors
    df_train['pSubjectSemanticErrors'] = p_semantic_errors

    return df_train


accuracy_list=[]
f1_score_list=[]
precision_score_list=[]
kappa_score_list=[]
recall_score_list=[]
tp=[]
fp=[]
fn=[]
tn=[]


frames=[]


for i in range(10):
    print("Fold=\t",i)
    print("\n")

    df_train=pd.read_csv("CV/Fold"+ str(i) +  "/Training.csv")
    df_test =pd.read_csv("CV/Fold"  + str(i) + "/Test.csv")

    df_train=pd.merge(df_merged_code,df_train,on=["StartOrder","SubjectID","ProblemID"])
    df_test=pd.merge(df_merged_code,df_test,on=["StartOrder","SubjectID","ProblemID"])

    df_train = df_train.replace(np.nan, '', regex=True)
    df_test = df_test.replace(np.nan, '', regex=True)

    df_pcorrect=df_train.groupby("ProblemID",as_index=False)["FirstCorrect"].mean()
    df_pcorrect=df_pcorrect.rename(columns={"FirstCorrect" : "Pcorrectforproblem"})
    df_train=pd.merge(df_pcorrect,df_train,on=["ProblemID"])


    df_pmedian = df_train.groupby("ProblemID",as_index=False)["Attempts"].median()
    df_pmedian=df_pmedian.rename(columns = {"Attempts" : "Pmedian" })
    df_train=pd.merge(df_pmedian,df_train,on=["ProblemID"])

    df_train=add_features_basic(df_train)
    df_test = add_features_basic(df_test)

    c = []
    dic = {}

    for index, rows in df_train.iterrows():
        _id = rows['ProblemID']
        if(_id in dic.keys()):
            c.append(dic[_id])
        else:
            d = df_train[df_train['ProblemID']==_id]
            f = len(d[d['is_semantic_error']==True].index)
            t = len(d.index)

            dic[_id] = (f*1.0)/t
            c.append((f*1.0)/t)

    df_train['pProblemSemanticError'] = c

    

    df_prob_synt=df_train.groupby("ProblemID",as_index=False)["is_syntax_error"].mean()
    df_prob_synt=df_prob_synt.rename(columns={"is_syntax_error" : "Prob_synt"})
    df_train=pd.merge(df_prob_synt,df_train,on=["ProblemID"])

    # print(df_test.shape)
    # print("log")
    df_temp= df_train[["ProblemID","Prob_synt","pProblemSemanticError","Pcorrectforproblem","Pmedian"]]

    l1 = []
    l2 = []
    l3 = []
    l4 = []

    for index, rows in df_test.iterrows():
        d = df_temp[df_temp['ProblemID']==rows['ProblemID']]
        # print(d[0:1]['Prob_synt'])
        l1.append(d.iloc[0, 1])
        l2.append(d.iloc[0, 2])
        l3.append(d.iloc[0, 3])
        l4.append(d.iloc[0, 4])

    df_test['Prob_synt'] = l1
    df_test['pProblemSemanticError'] = l2
    df_test['Pcorrectforproblem'] = l3
    df_test['Pmedian'] = l4

    # print(df_train[df_train["is_syntax_error"] == True].shape[0])
    # print(df_test[df_test["is_syntax_error"] == True].shape[0])

    print("\n")
    df_test_syntax_error = df_test[df_test["is_syntax_error"]== True]
    # df_test_fname_error = df_test[df_test["has_fname_error"]==True]

    df_train =df_train[df_train["is_syntax_error"]== False]
    df_test = df_test[df_test["is_syntax_error"]== False]



    df_train = df_train[["Pcorrectforproblem","Pmedian","p_prior_correct","p_prior_completed","prior_attempts","pProblemSemanticError","Prob_synt"
                        ,"pSubjectSyntaxErrors","pSubjectSemanticErrors","Correct" , "StartOrder","ProblemID","Code"]]
    df_test = df_test[["Pcorrectforproblem","Pmedian","p_prior_correct","p_prior_completed","prior_attempts","pProblemSemanticError","Prob_synt"
                        ,"pSubjectSyntaxErrors","pSubjectSemanticErrors","Correct","StartOrder" , "ProblemID", "Code"]]
    df_test_syntax_error = df_test_syntax_error[["Pcorrectforproblem","Pmedian","p_prior_correct","p_prior_completed","prior_attempts","pProblemSemanticError","Prob_synt"
                        ,"pSubjectSyntaxErrors","pSubjectSemanticErrors","Correct","StartOrder" , "ProblemID", "Code"]]



    frames.append(df_test)
    model = basic_model(df_train)
    pickle.dump(model, open(("model"+str(i) + ".pkl"),"wb"))

    test_x=df_test.iloc[:,input_feat]
    test_x=test_x.values
    test_y = df_test.iloc[:,output_feat]
    test_y =test_y.values

    prediction =  model.predict(test_x)
    df_test["prediction"] = prediction
    fold_id = [i]*len(prediction)
    df_test["Fold"] =fold_id

    prediction = list(prediction)
    test_synt_list=[False]*df_test_syntax_error.shape[0]

    df_test_syntax_error["prediction"] = test_synt_list
    fold_id = [i]*len(test_synt_list)
    df_test_syntax_error["Fold"] =fold_id

    prediction+=test_synt_list

    test_y = [i[0] for i in test_y]

    test_y+=list(df_test_syntax_error['Correct'])

    accuracy= accuracy_score(test_y, prediction)
    accuracy_list.append(accuracy)

    f1=f1_score(test_y,prediction)
    f1_score_list.append(f1)

    precision = precision_score(test_y,prediction)
    precision_score_list.append(precision)

    kappa=cohen_kappa_score(test_y,prediction)
    kappa_score_list.append(kappa)

    recall = recall_score(test_y,prediction)
    recall_score_list.append(recall)

    cm=confusion_matrix(test_y, prediction)
    tp.append(cm[0][0])
    fp.append(cm[0][1])
    fn.append(cm[1][0])
    tn.append(cm[1][1])

result=pd.concat(frames)
result.to_csv("cv_predict.csv",index = False)

d={"accuracy" :[mean(accuracy_list)]  , "f1_score" :mean(f1_score_list) , "precision_score" : mean(precision_score_list)  , "kappa_score" : mean(kappa_score_list) ,"recall_score" :mean(recall_score_list) }
df=pd.DataFrame(data=d)

df.to_csv("evaluation_overall.csv",index = False)


df=pd.read_csv("cv_predict.csv")
df=df[["ProblemID","Correct", "prediction"]]

col=[]
tn=[]
tp=[]
fn=[]
fp=[]

for index,rows in df.iterrows():
    if(rows["Correct"] == True  and rows["prediction"]==True):
        tp.append(1)
    else:
        tp.append(0)
    if(rows["Correct"] == False  and rows["prediction"]==True):
        fp.append(1)
    else:
        fp.append(0)
    if(rows["Correct"] == False and rows["prediction"]== False):
        tn.append(1)
    else:
        tn.append(0)
    if(rows["Correct"] == True  and rows["prediction"]== False):
        fn.append(1)
    else:
        fn.append(0)
        
    if(rows["Correct"] == rows["prediction"]):
        col.append(1)
    else:
        col.append(0)

    
df["accuracy"]=col
df["tp"] = tp
df["tn"] = tn
df["fp"] = fp
df["fn"] = fn


df_accuracy=df.groupby("ProblemID",as_index=False)["accuracy"].mean()
# df=pd.merge(df,df_accuracy,on=["ProblemID", "" ])
df_tn  =df.groupby("ProblemID",as_index=False)["tn"].mean()
df_tp  =df.groupby("ProblemID",as_index=False)["tp"].mean()
df_fn  =df.groupby("ProblemID",as_index=False)["fn"].mean()
df_fp  =df.groupby("ProblemID",as_index=False)["fp"].mean()
df_pcorrect=df.groupby("ProblemID",as_index=False)["Correct"].mean()
df_ppredicted = df.groupby("ProblemID" ,as_index=False)["prediction"].mean()


accuracy=list(df_accuracy["accuracy"])
tp=list(df_tp["tp"])
tn=list(df_tn["tn"])
fp=list(df_fp["fp"])
fn=list(df_fn["fn"])
pcorrect=list(df_pcorrect["Correct"])
ppredicted=list(df_ppredicted["prediction"])


df_accuracy["tp"] = tp
df_accuracy["tn"] = tn
df_accuracy["fp"] = fp
df_accuracy["fn"] = fn
df_accuracy["pcorrect"] = pcorrect
df_accuracy["ppredicted"] = ppredicted


df_test =df_accuracy
df_accuracy=df_accuracy.assign( precision = ( df_accuracy["tp"] )/ (df_accuracy["tp"] + df_accuracy["fp"]))
df_accuracy=df_accuracy.assign( recall  = ( df_accuracy["tp"] )/ (df_accuracy["tp"] + df_accuracy["fn"]))

df_accuracy=df_accuracy.assign( f1_score = ( 2*df_accuracy["precision"] *df_accuracy["recall"] )/ (df_accuracy["precision"] + df_accuracy["recall"]))
df_accuracy=df_accuracy.assign( pe = ( df_accuracy["pcorrect"] * df_accuracy["ppredicted"] ) + (1-df_accuracy["pcorrect"])*(1-df_accuracy["ppredicted"] ))
df_accuracy=df_accuracy.assign( kappa = ( df_accuracy["accuracy"] - df_accuracy["pe"])/ (1- df_accuracy["pe"]))


df_accuracy.to_csv("evaluation_by_problem.csv")