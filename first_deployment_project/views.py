from django.http import HttpRequest, HttpResponse
from django.shortcuts import render # used for rendering html

# importing model
import joblib

# def home(request):
#     return HttpResponse("The first page of the Model Deployment Page")   ### just some text on the first page 

def home(request):
    return render(request, 'home_page.html')

# view (form) for  model's prediction result
def result(request):
    model = joblib.load('Credit_Score_Classification_model.sav')   

    lst = []
    lst.append(request.GET['First_Attribute'])
    lst.append(request.GET['Second_Attribute'])
    lst.append(request.GET['Third_Attribute']) 
    lst.append(request.GET['Fourth_Attribute'])
    lst.append(request.GET['Fifth_Attribute'])
    lst.append(request.GET['Sixth_Attribute'])
    lst.append(request.GET['Seventh_Attribute'])
    lst.append(request.GET['Eighth_Attribute'])
    lst.append(request.GET['Ninth_Attribute'])
    
    print(f"Lst of all features: {lst}")

    prediction = model.predict([lst])

    class_str = 'Bad' if 1 in prediction else 'Standard' if 2 in prediction else 'Good'
    return render(request, 'result.html', {'ans': prediction, 'lst':lst, 'class_str': class_str})