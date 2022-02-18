from django.shortcuts import redirect
from . import views
from django.contrib.auth import authenticate, login,logout
from django.urls import path, include

urlpatterns = [
    path('', views.home,name='home-page'),
   path('design', views.design,name='result-page'),
path('output', views.output,name='output-page'),
path('newpage',  views.new_page,  name="my_function")

]