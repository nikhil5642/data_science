from rest_framework import serializers
from .models import stock

class StockSerializer(serializers.ModelSerializer):
   
      class Meta:
            model=stock
            fields='__all__' 
