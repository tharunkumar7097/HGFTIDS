
from django.shortcuts import render
from sklearn.model_selection import train_test_split

from collections import deque

WINDOW = 2
sensor_buffer = deque(maxlen=WINDOW)


from .models import userRegisteredTable
from django.core.exceptions import ValidationError
from django.contrib import messages


def userRegisterCheck(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        username = request.POST.get("loginId")
        mobile = request.POST.get("mobile")
        password = request.POST.get("password")
        

        # Create an instance of the model
        user = userRegisteredTable(
            name=name,
            email=email,
            loginid=username,
            mobile=mobile,
            password=password,
            
        )

        try:
            # Validate using model field validators
            user.full_clean()
            
            # Save to DB
            user.save()
            messages.success(request,'registration Successfully done,please wait for admin APPROVAL')
            return render(request, "userRegisterForm.html")


        except ValidationError as ve:
            # Get a list of error messages to display
            error_messages = []
            for field, errors in ve.message_dict.items():
                for error in errors:
                    error_messages.append(f"{field.capitalize()}: {error}")
            return render(request, "userRegisterForm.html", {"messages": error_messages})

        except Exception as e:
            # Handle other exceptions (like unique constraint fails)
            return render(request, "userRegisterForm.html", {"messages": [str(e)]})

    return render(request, "userRegisterForm.html")


def userLoginCheck(request):
    if request.method=='POST':
        username=request.POST['userUsername']
        password=request.POST['userPassword']

        try:
            user=userRegisteredTable.objects.get(loginid=username,password=password)

            if user.status=='Active':
                request.session['id']=user.id
                request.session['name']=user.name
                request.session['email']=user.email
                
                return render(request,'users/userHome.html')
            else:
                messages.error(request,'Status not activated please wait for admin approval')
                return render(request,'userLoginForm.html')
        except:
            messages.error(request,'Invalid details please enter details carefully or Please Register')
            return render(request,'userLoginForm.html')
    return render(request,'userLoginForm.html')


def userHome(request):
    if not request.session.get('id'):
        return render(request,'userLoginForm.html')
    return render(request,'users/userHome.html')



import pandas as pd
import numpy as np
import joblib
from Users.training import gnn_training
def training(request):
        
    # b=gnn_training()
        
    results1=pd.read_csv(r'media\metrics.csv')
    dff = results1.to_html(classes='table table-striped', index=False) 
        # Pass DataFrame to template (convert to dict for easier rendering)
    return render(request, 'users/training.html', {
            
        'results_df':dff  # Convert DataFrame to list of dictionaries
        })

import os
import joblib
import numpy as np
import pandas as pd

from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect

NUM_CHANNELS = 8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained components
model = joblib.load(os.path.join(BASE_DIR, "media", "hgft_ids_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "media", "scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "media", "label_encoder.pkl"))

@csrf_protect
def prediction(request):

    result = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            values = [
                float(request.POST[f"data_{i}"])
                for i in range(8)
            ]

            df = pd.DataFrame(
                [values],
                columns=[f"DATA_{i}" for i in range(8)]
            )

            X_scaled = scaler.transform(df)

            pred_encoded = model.predict(X_scaled)[0]
            probs = model.predict_proba(X_scaled)[0]

            confidence = round(float(np.max(probs)) * 100, 2)

            result = label_encoder.inverse_transform([pred_encoded])[0]

        except Exception as e:
            error = str(e)

    return render(request, "users/prediction.html", {
        "result": result,
        "confidence": confidence,
        "error": error
    })
