from django.shortcuts import render
from django.views.decorators.cache import cache_control
# Create your views here.
from django.shortcuts import render
from django.contrib import messages

from Users.models import userRegisteredTable

@cache_control(no_cache=True,must_revalidate=True,no_store=True)
def adminHome(request):
    if not request.session.get('admin'):
        return render(request,'adminLoginForm.html')  
    return render(request, 'admin/adminHome.html')

def adminLoginCheck(request):
    if request.method=="POST":
        username=request.POST['adminUsername']
        adminPassword=request.POST['adminPassword']

        if adminPassword=="admin" and username=='admin':
            request.session['admin']=True
            return render(request,'admin/adminHome.html')
        else:
            messages.error(request,'Invalid details')
            return render(request,'adminLoginForm.html')
    else:
        return render(request,'adminLoginForm.html')
    

def userList(request):
    if not request.session.get('admin'):
        return render(request,'adminLoginForm.html') 
    users=userRegisteredTable.objects.all()
    return render(request,'admin/userList.html',{'users':users})


def log(request):
    request.session.flush()  # clears all session data
    return render(request,'index.html')


from django.shortcuts import redirect, get_object_or_404
from django.contrib import messages

def activate_user(request):
    if not request.session.get('admin'):
        return render(request,'adminLoginForm.html') 

    id=request.GET['id']
    user = get_object_or_404(userRegisteredTable, id=id)
    user.status = 'Active'
    user.save()
    users=userRegisteredTable.objects.all()
    return render(request,'admin/userList.html',{'users':users})  

def deactivate_user(request):
    if not request.session.get('admin'):
        return render(request,'adminLoginForm.html') 
    
    id=request.GET['id']
    user = get_object_or_404(userRegisteredTable, id=id)
    user.status = 'Inactive'
    user.save()
    users=userRegisteredTable.objects.all()
    return render(request,'admin/userList.html',{'users':users})

