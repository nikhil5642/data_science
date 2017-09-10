from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views  import APIView
from .serializers import StockSerializer
from .models import stock

'''def index(request):
    return HttpResponse("Hello, world. You're at the api index.")
'''
class StockList(APIView):
    
      def get(self,request): 
          stocks=stock.objects.all()
          serializer=StockSerializer(stocks,many=True)
          return Response(serializer.data)
      
      def post(self):  
          pass
