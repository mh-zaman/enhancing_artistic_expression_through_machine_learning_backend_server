from django.urls import path
from api_server import views

urlpatterns = [
    path('', views.check, name='check'),
    path('predict', views.predict, name='predict'),
    path('predict/', views.predict, name='predict'),
]
