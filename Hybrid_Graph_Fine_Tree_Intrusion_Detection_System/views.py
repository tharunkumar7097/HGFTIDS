


from django.shortcuts import render


def index(request):
    return render(request,'index.html')


def adminLoginForm(request):
    return render(request,'adminLoginForm.html')


def userLoginForm(request):
    return render(request,'userLoginForm.html')


def userRegisterForm(request):
    return render(request,'userRegisterForm.html')