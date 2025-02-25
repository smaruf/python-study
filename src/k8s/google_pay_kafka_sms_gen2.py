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

def setup_project_structure(base_path, app_package):
    """ Sets up the entire project structure with necessary directories and files. """
    # Convert Java package to path
    package_path = app_package.replace('.', '/')
    
    # Backend paths
    backend_path = os.path.join(base_path, 'backend')
    src_main_path = os.path.join(backend_path, 'src/main')
    src_test_path = os.path.join(backend_path, 'src/test')
    java_path = os.path.join(src_main_path, 'java', package_path)
    resources_path = os.path.join(src_main_path, 'resources')
    test_java_path = os.path.join(src_test_path, 'java', package_path)

    # Infrastructure paths
    infra_path = os.path.join(base_path, 'infra')
    terraform_path = os.path.join(infra_path, 'terraform')
    k8s_path = os.path.join(infra_path, 'k8s')
    
    # Create backend directories
    for path in [java_path, resources_path, test_java_path]:
        create_directory(path)

    # Create BDD and E2E test directories
    bdd_test_path = os.path.join(test_java_path, 'bdd')
    e2e_test_path = os.path.join(test_java_path, 'e2e')
    create_directory(bdd_test_path)
    create_directory(e2e_test_path)
    create_directory(os.path.join(bdd_test_path, 'stepdefs'))
    create_directory(os.path.join(resources_path, 'features'))

    # Create infrastructure directories
    for path in [terraform_path, k8s_path]:
        create_directory(path)
        
    # Create starter files in backend
    application_java_content = f"""\
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
    create_file(os.path.join(java_path, 'Application.java'), application_java_content)
    create_file(os.path.join(resources_path, 'application.properties'), '# Spring configuration properties')

    # Starter BDD test integration with Cucumber
    cucumber_java_content = f"""\
package {app_package}.bdd;

import io.cucumber.junit.Cucumber;
import io.cucumber.junit.CucumberOptions;
import org.junit.runner.RunWith;

@RunWith(Cucumber.class)
@CucumberOptions(
    features = "classpath:features",
    glue = "{app_package}.bdd.stepdefs"
)
public class CucumberIntegrationTest {{
}}
"""
    create_file(os.path.join(bdd_test_path, 'CucumberIntegrationTest.java'), cucumber_java_content)
    
    # Starter E2E testing with Selenium
    selenium_java_content = f"""\
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
    create_file(os.path.join(e2e_test_path, 'SeleniumTests.java'), selenium_java_content)

    # Terraform and Kubernetes starter files for Infrastructure
    main_tf_content = '# Example Terraform configuration\n'
    create_file(os.path.join(terraform_path, 'main.tf'), main_tf_content)
    k8s_deployment_content = '# Example Kubernetes deployment\n'
    k8s_service_content = '# Example Kubernetes service\n'
    create_file(os.path.join(k8s_path, 'deployment.yaml'), k8s_deployment_content)
    create_file(os.path.join(k8s_path, 'service.yaml'), k8s_service_content)

    # Maven pom.xml including essential dependencies
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
        <spring-cloud.version>Hoxton.SR3</spring-cloud.version>
    </properties>

    <dependencies>
        <!-- Spring Boot -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <!-- Spring Kafka -->
        <dependency>
            <groupId>org.springframework.kafka</groupId>
            <artifactId>spring-kafka</artifactId>
        </dependency>
        <!-- AWS Java SDK SNS -->
        <dependency>
            <groupId>com.amazonaws</groupId>
            <artifactId>aws-java-sdk-sns</artifactId>
        </dependency>
        <!-- Spring Data DynamoDB (or other Spring Data project as necessary) -->
        <dependency>
            <groupId>com.github.derjust</groupId>
            <artifactId>spring-data-dynamodb</artifactId>
            <version>5.1.0</version>
        </dependency>
        <!-- Required if working with REST Docs -->
        <dependency>
            <groupId>org.springframework.restdocs</groupId>
            <artifactId>spring-restdocs-mockmvc</artifactId>
            <scope>test</scope>
        </dependency>
        <!-- Cucumber for BDD -->
        <dependency>
            <groupId>io.c](maven gen dependen)ucumber.dat</groupId>
            <artifactId>cucumber-spring</artifactId>
            <version>6.8.1</version>
            <scope>test</scope>
        </dependency>
        <!-- Selenium for E2E Testing -->
        <dependency>
            <groupId>org.seleniumhq.selenium</groupId>
            <artifactId>selenium-java</artifactId>
            <version>3.141.59</version>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.cloud</groupId>
                <artifactId>spring-cloud-dependencies</artifactId>
                <version>${{spring-cloud.version}}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>
"""
    create_file(os.path.join(backend_path, 'pom.xml'), pom_xml_content)
    create_file(os.path.join(base_path, 'README.md'), '# Project Documentation')

def main():
    project_name = input("Enter the project name: ")
    base_path = os.path.abspath(input("Enter the destination directory path: "))
    project_path = os.path.join(base_path, project_name)
    app_package = 'com.example.backend'
    
    print("Setting up the project structure...")
    setup_project_structure(project_path, app_package)
    print("Project setup completed successfully.")

if __name__ == '__main__':
    main()
