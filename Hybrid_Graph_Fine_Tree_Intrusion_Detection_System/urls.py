
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from . import views as mv
from Admin import views as av
from Users import  views as uv
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',mv.index,name='index'),
    path('adminLoginForm',mv.adminLoginForm,name='adminLoginForm'),
    path('userLoginForm',mv.userLoginForm,name='userLoginForm'),
    path('userRegisterForm',mv.userRegisterForm,name='userRegisterForm'),


    path('adminLoginCheck',av.adminLoginCheck,name='adminLoginCheck'),
    path('adminHome',av.adminHome,name='adminHome'),
    path('userList',av.userList,name='userList'),
    path('activate_user',av.activate_user,name='activate_user'),
    path('deactivate_user',av.deactivate_user,name='deactivate_user'),
    
    path('log',av.log,name='log'),






    path('userRegisterCheck',uv.userRegisterCheck,name='userRegisterCheck'),
    path('userLoginCheck',uv.userLoginCheck,name='userLoginCheck'),
    path('userHome',uv.userHome,name='userHome'),
    path('training',uv.training,name='training'),
    path('prediction',uv.prediction,name='prediction'),

    

    
    
    
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

