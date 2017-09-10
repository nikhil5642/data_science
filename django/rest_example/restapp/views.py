from rest_framework.decorators import api_view
from django.contrib.auth.models import User
from django.http import Http404

from restapp.serializer import UserSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

'''@api_view(["POST"])
def create_user(request):
    serializer = UserSerializer(request.data)
   
    if serializer.is_valid():
        serializer.save()
        return Response({"message": "User created"}) 
    else:
        data = {
          "error": True,
          "errors": serializer.errors,          
        }
        return Response(data) 






@api_view(["GET"])
def user_details(request, pk):
    user = User.objects.get(id=pk)
    serializer = UserSerializer(user)
    return Response(serializer.data)

@api_view(["GET", "PUT"])
def user_update(request, pk):
    user = User.objects.get(id=pk)
    if request.method == "PUT":
        serializer = UserSerializer(user, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        else:
            return Response({"error": serializer.errors, "error": True}) 
    serializer = UserSerializer(user)
    return Response(serializer.data)

'''
class UserList(APIView):

    def post(self, request, format=None):
        serializer = UserSerializer(data=request.DATA)
        raise Http404
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



    
    def delete(self, request, pk, format=None):
        user = self.get_object(pk)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class UserDetail(APIView):
    """
    Retrieve, update or delete a user instance.
    """
    def get_object(self, pk):
        try:
            return User.objects.get(pk=pk)
        except User.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        user = self.get_object(pk)
        user = UserSerializer(user)
        raise Http404
        return Response(user.data)

    def put(self, request, pk, format=None):
        user = self.get_object(pk)
        serializer = UserSerializer(user, data=request.DATA)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        user = self.get_object(pk)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

