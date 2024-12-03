import os
import sys
import grpc
from concurrent import futures
from django.core.management import execute_from_command_line
from django.conf import settings
from django.apps import AppConfig


# Define django settings
settings.configure(
    DEBUG=True,
    ROOT_URLCONF=__name__,
    INSTALLED_APPS=[
        'django.contrib.contenttypes',
        'django.contrib.auth',
        'django_grpc',
    ],
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': 'db.sqlite3',
        }
    },
    USE_TZ=True,
)


# Define URL patterns
from django.urls import path
from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello, world from Django!")


urlpatterns = [
    path('', index),
]


# Define gRPC services 
from grpc_framework import methods
from grpc_framework.services import Service


class CoreService(Service):
    @methods.unary_unary
    def SayHello(self, request, context):
        name = request.name
        return {'message': f'Hello, {name} from gRPC!'}


# Django App Config
class DjangoGrpcAppConfig(AppConfig):
    name = 'django_grpc'
    verbose_name = "Django gRPC Integration Example"

    def ready(self):
        # This function registers gRPC services when Django starts.
        from grpc_framework.server import server
        server.add_service(CoreService.as_servicer())


# Run as a script
if __name__ == '__main__':
    # Adding Django app to system path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # Prepare app configurations
    django_grpc_app_config = DjangoGrpcAppConfig('django_grpc', __name__)
    django_grpc_app_config.ready()

    # Decide if running as a Django HTTP server or gRPC server
    if 'runserver' in sys.argv:
        execute_from_command_line(sys.argv)
    elif 'grpcserver' in sys.argv:
        # Starting a gRPC server
        from grpc_framework.server import server
        server.start()  # Default is to listen on port 50051
    else:
        print("Usage: python django_grpc.py runserver | python django_grpc.py grpcserver")
