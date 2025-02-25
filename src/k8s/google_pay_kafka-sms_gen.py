import os

def create_directory(path):
    """ Ensures that the specified directory exists, creating it if necessary. """
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"Created directory: {path}")

def create_file(path, content):
    """ Creates a file with the specified content. """
    with open(path, 'w') as file:
        file.write(content)
    print(f"Created file: {path}")

def setup_backend(project_path, app_package):
    """ Sets up a Spring Boot application with several integrations and configurations. """
    package_path = app_package.replace('.', '/')
    java_path = os.path.join(project_path, 'backend/src/main/java', package_path)
    resources_path = os.path.join(project_path, 'backend/src/main/resources')
    test_path = os.path.join(project_path, 'backend/src/test/java', package_path)
    bdd_path = os.path.join(test_path, 'bdd')
    e2e_path = os.path.join(test_path, 'e2e')
    features_path = os.path.join(resources_path, 'features')

    directories = [java_path, resources_path, test_path, bdd_path, e2e_path, features_path]
    for path in directories:
        create_directory(path)

    # Main Application Java
    application_java = f"""\
package {app_package};

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {{
    public static void main(String[] args) {{
        SpringApplication.run(Application.class, args);
    }}
}}
"""
    create_file(os.path.join(java_path, 'Application.java'), application_java)

    # Maven pom.xml
    pom_xml_content = f"""\
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>{app_package}</groupId>
    <artifactId>backend</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <java.version>17</java.version>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
    </properties>

    <dependencies>
        <!-- Spring Boot Dependencies -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- Kafka -->
        <dependency>
            <groupId>org.springframework.kafka</groupId>
            <artifactId>spring-kafka</artifactId>
            <version>2.7.8</version>
        </dependency>
        
        <!-- AWS SNS for SMS -->
        <dependency>
            <groupId>com.amazonaws</groupId>
            <artifactId>aws-java-sdk-sns</artifactId>
            <version>1.12.118</version>
        </dependency>
        
        <!-- OpenAPI -->
        <dependency>
            <groupId>org.springdoc</groupId>
            <artifactId>springdoc-openapi-ui</artifactId>
            <version>1.6.4</version>
        </dependency>
        
        <!-- DynamoDB -->
        <dependency>
            <groupId>com.amazonaws</groupId>
            <artifactId>aws-java-sdk-dynamodb</artifactId>
            <version>1.12.118</version>
        </dependency>
        
        <!-- Cache Configuration -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-cache</artifactId>
        </dependency>

        <!-- AWS OpenSearch -->
        <dependency>
            <groupId>org.opensearch.client</groupId>
            <artifactId>opensearch-rest-high-level-client</artifactId>
            <version>1.1.0</version>
        </dependency>
        
        <!-- AWS Systems Manager Parameter Store -->
        <dependency>
            <groupId>com.amazonaws</groupId>
            <artifactId>aws-java-sdk-ssm</artifactId>
            <version>1.12.118</version>
        </dependency>
        
        <!-- Cucumber for BDD -->
        <dependency>
            <groupId>io.cucumber</groupId>
            <artifactId>cucumber-java</artifactId>
            <version>6.10.2</version>
            <scope>test</scope>
        </dependency>

        <dependency>
            <groupId>io.cucumber</groupId>
            <artifactId>cucumber-spring</artifactId>
            <version>6.10.2</version>
            <scope>test</scope>
        </dependency>

        <!-- Selenium for E2E tests -->
        <dependency>
            <groupId>org.seleniumhq.selenium</groupId"
            <artifactId>selenium-java</artifactId>
            <version>3.141.59</version>
            <scope>test</scope>
        </dependency>
        
    </dependencies>
</project>
"""
    create_file(os.path.join(project_path, 'backend/pom.xml'), pom_xml_content)

    # Cucumber Integration Test Setup
    cucumber_test_java = f"""\
package {app_package}.bdd;

import io.cucumber.junit.Cucumber;
import io.cucumber.junit.CucumberOptions;
import org.junit.runner.RunWith;

@RunWith(Cucumber.class)
@CucumberOptions(
    features = "src/test/resources/features",
    glue = "{app_package}.bdd.stepdefs"
)
public class CucumberIntegrationTest {{
}}
"""
    create_file(os.path.join(bdd_path, 'CucumberIntegrationTest.java'), cucumber_test_java)

    # Selenium Test Setup
    selenium_test_java = f"""\
package {app_package}.e2e;

import org.junit.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.firefox.FirefoxDriver;

public class SeleniumTests {{
    @Test
    public void testPageNavigation() {{
        System.setProperty("webdriver.gecko.driver", "/path/to/geckodriver");
        WebDriver driver = new FirefoxDriver();
        driver.get("http://www.example.com");
        driver.quit();
    }}
}}
"""
    create_file(os.path.join(e2e_path, 'SeleniumTests.java'), selenium_test_java)

    # application.properties for the backend module
    application_properties = """\
spring.kafka.bootstrap-servers=localhost:9092
spring.kafka.consumer.group-id=my-group
spring.kafka.template.default-topic=my-topic

# OpenAPI configuration
springdoc.api-docs.path=/api-docs
springdoc.swagger-ui.path=/swagger-ui.html
"""
    create_file(os.path.join(resources_path, 'application.properties'), application_properties)

def main():
    project_name = input("Enter the project name: ")
    destination = input("Enter the destination path (relative to the current folder): ")
    app_package = 'com.example.backend'

    project_path = os.path.join(os.getcwd(), destination, project_name)

    print("Setting up project directories and files...")
    setup_backend(project_path, app_package)

    print("Project setup completed successfully.")

if __name__ == '__main__':
    main()
