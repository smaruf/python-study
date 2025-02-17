import subprocess

def generate_terraform_file():
    content = """
provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region"
  default     = "us-west-2"
}

variable "bucket_name" {
  description = "S3 Bucket for assets"
  default     = "my-assets-bucket"
}

variable "s3_origin_identity" {
  description = "CloudFront Origin Access Identity"
  default     = "origin-access-identity/cloudfront/EXAMPLE"
}

module "s3_bucket" {
  source   = "./modules/s3_bucket"
  name     = var.bucket_name
}

module "cloudfront" {
  source   = "./modules/cloudfront"
  s3_bucket_id = module.s3_bucket.id
  origin_access_identity = var.s3_origin_identity
}

module "lambda_functions" {
  source = "./modules/lambda"
}

module "cloudwatch_logs" {
  source         = "./modules/cloudwatch"
  lambda_function_name = module.lambda_functions.function_name
}

output "cloudfront_domain_name" {
  value = module.cloudfront.domain_name
}
"""
    with open('main.tf', 'w') as file:
        file.write(content)
    print("Terraform configuration file 'main.tf' has been created.")

def write_module_files():
    modules = {
        "s3_bucket": """
resource "aws_s3_bucket" "bucket" {
  bucket = var.name
}

output "id" {
  value = aws_s3_bucket.bucket.id
}
""",
        "cloudfront": """
resource "aws_cloudfront_distribution" "distribution" {
  origin {
    domain_name = var.s3_bucket_id
    origin_id   = "S3-Origin"
    s3_origin_config {
      origin_access_identity = var.origin_access_identity
    }
  }

  enabled = true
  default_root_object = "index.html"
  
  # Additional configurations here

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

output "domain_name" {
  value = aws_cloudfront_distribution.distribution.domain_name
}
""",
        "lambda": """
# Lambda function configurations and outputs
""",
        "cloudwatch": """
resource "aws_cloudwatch_log_group" "log_group" {
  name = "/aws/lambda/${var.lambda_function_name}"
}

# Define alarms and other monitoring metrics
"""
    }
    for module_name, module_content in modules.items():
        directory = f"./modules/{module_name}"
        subprocess.run(["mkdir", "-p", directory])
        with open(f"{directory}/main.tf", "w") as file:
            file.write(module_content)
    
    print("Module files have been created.")

def run_terraform_commands():
    # Initializing Terraform
    subprocess.run(["terraform", "init"], check=True)
    
    # Applying Terraform configuration
    subprocess.run(["terraform", "apply", "-auto-approve"], check=True)

def main():
    generate_terraform_file()
    write_module_files()
    run_terraform_commands()

if __name__ == '__main__':
    main()
