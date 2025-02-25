## Google pay project structure:
```
project_root/
│
├── backend/                                                 # Backend Spring Boot Application
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/example/backend/   
│   │   │   │   ├── Application.java                         # Main spring application
│   │   │   │   ├── config/                                  # Configuration for Kafka, DynamoDB, SMS, etc.
│   │   │   │   │   ├── KafkaConfig.java
│   │   │   │   │   ├── DynamoDBConfig.java
│   │   │   │   │   ├── SnsConfig.java
│   │   │   │   │   └── OpenSearchConfig.java
│   │   │   │   ├── controller/                              # REST API Controllers
│   │   │   │   ├── service/                                 # Service layer for business logic
│   │   │   │   ├── repository/                              # Repositories (if using Spring Data)
│   │   │   │   └── model/                                   # Entity models
│   │   │   └── resources/
│   │   │       ├── application.properties                   # Application settings
│   │   │       └── features/                                # Cucumber features folder
│   │   ├── test/
│   │   │   ├── java/com/example/backend/
│   │   │   │   ├── bdd/
│   │   │   │   │   ├── CucumberIntegrationTest.java
│   │   │   │   │   └── stepdefs/                            # BDD Step definitions
│   │   │   │   └── e2e/
│   │   │   │       └── SeleniumTests.java                   # Selenium E2E tests
│   │   └── resources/
│   │       └── application-test.properties                  # Testing-specific properties
│   └── pom.xml                                              # Maven configuration file
│
├── infra/                                                   # Infrastructure as Code (Terraform and Kubernetes configs)
│   ├── terraform/                                           # Terraform configuration files
│   │   └── main.tf                                          # Main terraform file
│   └── k8s/                                                 # Kubernetes manifests
│       ├── deployment.yaml                                  # Deployment config
│       └── service.yaml                                     # Service config
│
├── .mvn/                                                    # Maven wrapper (to ensure version consistency)
│   └── wrapper/
│       ├── MavenWrapperDownloader.java
│       ├── maven-wrapper.jar
│       └── maven-wrapper.properties
│
├── mvnw
├── mvnw.cmd
└── README.md                                                # Project documentation
```
