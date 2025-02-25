import os

def create_directory(path):
    """Create directory if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def create_file(path, content):
    """Create a file with given content."""
    with open(path, 'w') as file:
        file.write(content)
    print(f"Created file: {path}")

def setup_backend():
    """Set up backend Spring Boot application with Java 17."""
    base_path = 'backend'
    java_path = os.path.join(base_path, 'src/main/java/com/example/backend')
    resources_path = os.path.join(base_path, 'src/main/resources')
    test_path = os.path.join(base_path, 'src/test/java/com/example/backend')
    
    create_directory(base_path)
    create_directory(os.path.join(base_path, 'src/main/java'))
    create_directory(java_path)
    create_directory(resources_path)
    create_directory(os.path.join(base_path, 'src/test/java'))
    create_directory(test_path)

    # Creating pom.xml with Java 17 configuration
    pom_content = """<project xmlns="http://maven.apache.org/POM/4.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>backend</artifactId>
    <version>1.0.0</version>
    <properties>
        <java.version>17</java.version>
        <spring-boot.version>2.5.5</spring-boot.version>
    </properties>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>${spring-boot.version}</version>
    </parent>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
    </dependencies>

    </project>
    """
    create_file(os.path.join(base_path, 'pom.xml'), pom_content)

    # Spring Boot Main Application
    main_class_content = """package com.example.backend;
    
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
    """
    create_file(os.path.join(java_path, 'Application.java'), main_class_content)

def setup_lambda_python():
    """Setup Python AWS Lambda handlers."""
    lambda_path = 'lambda'
    create_directory(lambda_path)
    
    token_creator = """import json
import jwt

def lambda_handler(event, context):
    token = jwt.encode({'user': 'example'}, 'secret', algorithm='HS256')
    return {
        'statusCode': 200,
        'body': json.dumps({'token': token})
    }
    """
    create_file(os.path.join(lambda_path, 'token_creator.py'), token_creator)

def setup_kubernetes():
    """Setup Kubernetes configurations."""
    k8s_path = 'k8s'
    create_directory(k8s_path)
    
    backend_deployment = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: myregistry/backend:latest
        ports:
        - containerPort: 8080
    """
    create_file(os.path.join(k8s_path, 'backend_deployment.yaml'), backend_deployment)

def setup_terraform():
    """Setup Terraform for AWS resources."""
    terraform_path = 'terraform'
    create_directory(terraform_path)
    
    main_tf = """terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.0"
    }
  }

  provider "aws" {
    region  = "us-east-1"
  }
}

resource "aws_dynamodb_table" "example_table" {
  name           = "ExampleTable"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "Id"

  attribute {
    name = "Id"
    type = "S"
  }
}
    """
    create_file(os.path.join(terraform_path, 'main.tf'), main_tf)

def setup_cicd():
    """Setup basics for CI/CD using Jenkinsfile."""
    ci_cd_path = 'ci_cd'
    create_directory(ci_cd_path)
    
    jenkinsfile = """pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'mvn clean install'
            }
        }
    }
}
    """
    create_file(os.path.join(ci_cd_path, 'Jenkinsfile'), jenkinsfile)

if __name__ == '__main__':
    setup_backend()
    setup_lambda_python()
    setup_kubernetes()
    setup_terraform()
    setup_cicd()
    print("Complete advanced project setup with Java, Python, Kubernetes, Terraform, and CI/CD setup completed.")
