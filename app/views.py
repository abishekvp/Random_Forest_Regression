from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages

# regression imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import xgboost as xg
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

regression_name = {"reg_name":""}

def choose_regression(reg_name):
    if reg_name == "Random Forest Regression":return RandomForestRegressor(n_estimators = 10, random_state = 0)
    
    elif reg_name == "Linear Regression":return linear_model.LinearRegression()

    elif reg_name == "XGB Regression":return xg.XGBRegressor(n_estimators = 10, seed = 123)
    
    elif reg_name == "SVR Regression":return SVR(kernel='rbf')
    
    elif reg_name == "DTR Regression":return DT(random_state=0)
    
    elif reg_name == "ADA Regression":return AdaBoostRegressor()
    
    elif reg_name == "Cat Booste Regression":return CatBoostRegressor()
    
    elif reg_name == "Light GBM Regression":return LGBMRegressor()
        
    elif reg_name == "Bayesian Regression":return BayesianRidge()
    
    elif reg_name == "Lasso Regression":return Lasso()
    
    elif reg_name == "Ridge Regression":return Ridge()
    
    elif reg_name == "Elastic Net Regression":return ElasticNet()


def regression(request):

    total_dict = {}
    data_dict = {}
    table_data = {}
    subacc=[]

    def loaddata(i):
        url="static/csv/{0}.csv"
        data=pd.read_csv(url.format(i))
        return data


    def sepdf(df):
        x=df.iloc[:,4:7]
        y=df.iloc[:,7:8]
        return x,y,df

    def returnGrade(m):
        m=int(m)
        if(m>=91 and m<=100):
            return 10
        elif(m>=81 and m<=90):
            return 9
        elif(m>=71 and m<=80):
            return 8
        elif(m>=61 and m<=70):
            return 7
        elif(m>=50 and m<=60):
            return 6
        else:
            return 0

    def predGrade(df):
        grade=[]
        for i in range(len(df)):
            m=returnGrade(df.iloc[i,5])
            grade.append(m)
        gra=pd.DataFrame(grade)
        return gra


    def modeleval(y_test,Y_pred,model,name,df,sub):
        x,y,df=sepdf(df)

        data_dict.update({"title4":"{} REGRESSION evaluation of the model on unseen data".format(name)})
        MAE=metrics.mean_absolute_error(y_test,Y_pred)
        data_dict.update({"mae":"1.MAE: "+str(MAE)})
        MSE=metrics.mean_squared_error(y_test,Y_pred)
        data_dict.update({"mse":"2.MSE: "+str(MSE)})
        RMSE=np.sqrt(metrics.mean_absolute_error(y_test,Y_pred))
        data_dict.update({"rmse":"3.RMSE "+str(RMSE)})
        R2=metrics.r2_score(y_test,Y_pred)
        data_dict.update({"good_fit":"4.Goodness of fit(R2) of unseen data: "+str(R2)})
        n=len(x)
        k=3
        adjR2=1-((1-R2)*(n-1)/(n-k-1))
        data_dict.update({"adjusted_sqr":"5.Adjusted R-Square: "+str(adjR2)})
        data_dict.update({"variance_score":"Variance score: {}".format(model.score(x,y))})
        
        data_dict.update({"title5":"PREDICTED VALUE SEM MARKS FOR ALL THE STUDENT IN THE DATASET BASED ON IA1,IA2,MODEL EXAM MARKS"})
        predALL=model.predict(x) 
        predALL_pd=pd.DataFrame(predALL)
        df["Prediction"]=predALL_pd
        col=['Student Name','IAT 1','IAT 2','Model exam',"SemMark","Prediction","Actual Grade"]
        new_df=pd.DataFrame()
        
        
        for j in range(len(col)):
            new_df[col[j]]=df[col[j]]
            
        new_df["PREDGRADE"]=predGrade(new_df)
        CorW,correct,wrong=checkPredGrade(new_df)
        data_dict.update({"crt_pred":"Correct Prediction : "+str(correct), "wrong_pred":"Wrong Prediction : "+str(wrong)})
        new_df["Pred_label"]=CorW
        
        accuracy=int((correct/len(new_df))*100)
        data_dict.update({"accuracy":"Accuracy: "+str(accuracy)+"%"})
        subacc.append(accuracy)
        
        student_name = new_df['Student Name']
        iat_1 = new_df['IAT 1']
        iat_2 = new_df['IAT 2']
        model_exam = new_df['Model exam']
        semmark = new_df["SemMark"]
        prediction = new_df["Prediction"]
        actual_grade = new_df["Actual Grade"]
        predgrade = new_df["PREDGRADE"]
        
        for i in range(len(student_name)):
            data = {"iat_1":iat_1[i], "iat_2":iat_2[i], "model_exam":model_exam[i], "semmark":semmark[i], "prediction":prediction[i], "actual_grade":actual_grade[i], "predgrade":predgrade[i]}
            table_data.update({student_name[i]:data})        
        

    def checkPredGrade(df):
        count=0
        CorW=[]
        for i in range(len(df)):
            if(df.iloc[i,6]==df.iloc[i,7]):
                count+=1
                CorW.append(1)
            else:
                CorW.append(0)
        CorW_df=pd.DataFrame(CorW)
        wrong=len(df)-count
        return CorW_df,count,wrong
            

    def Regress(x,y,subject,reg_name):
        X_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
        Regression = choose_regression(reg_name)
        Regression.fit(X_train, y_train.values.ravel())
        
        y_pred_train = Regression.predict(X_train)
        r2_train=r2_score(y_train, y_pred_train)
        data_dict.update({"subject":subject})
        data_dict.update({"title1":reg_name+" REGRESSION evaluation of the model on training data"})
        data_dict.update({"title2":"training data r2 score:"+str(r2_train)})
        
        Y_pred=Regression.predict(x_test)
        modeleval(y_test,Y_pred,Regression,reg_name,df,subject)
        return Regression


    subject=["DM","DPS","DS","OOP","CA","CE"]
    fullform={"DM":"Discrete Mathematice","DPS":"Digital Principals and System Design","DS":"Data Structure","OOP":"Object Oriented Programming"
            ,"CA":"Computer Architecture","CE":"Communication Engineering"}

    def avgacc(subacc):
        sums=0
        for i in range(len(subacc)):
            sums+=subacc[i]
        #Average accuracy for each subject.
        average=sums/len(subacc)
        return average

    for i in range(len(subject)):
        data_dict = {}
        table_data = {}
        df=loaddata(subject[i])
        x,y,df=sepdf(df)
        sub_fullname=fullform[subject[i]]
        Regress(x,y,sub_fullname,regression_name.get("reg_name"))
        total_dict.update({str(subject[i]):{"data_dict":data_dict, "table_data":table_data}})
    
    average=avgacc(subacc)
    total_dict.update({"avg_accuracy":"Average Accuracy : "+str(average)+"%"})
    
    return render(request, 'regression.html', {"total_dict":total_dict})


def index(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            r_name = request.POST.get('regression_model')
            if r_name is not None:
                regression_name.update({"reg_name":r_name})
                return redirect("regression")
            else:messages.info(request, 'Select Regression Model')
        return render(request,'index.html')
    else:
        return redirect('signin')

def signin(request):
    if request.user.is_authenticated:return redirect('index')
    elif request.method == 'POST':    
        username = str(request.POST["username"]).lower()
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:login(request, user)
        else:messages.info(request, 'User not found')
    return render(request,'signin.html')

def signup(request):
    if request.user.is_authenticated:return redirect('index')
    elif request.method == 'POST':
        username = str(request.POST['username']).lower()
        email = request.POST['email']
        password = request.POST['password']
        if User.objects.filter(username=username).exists()==False:
            User.objects.create_user(username, email, password)
            user = authenticate(request, username=username, password=password)
            login(request, user)
        else:messages.info(request, 'User already exists')
        return redirect("index")
    return render(request,'signup.html')

def signout(request):
    if request.user.is_authenticated:
        logout(request)
        return redirect('signin')