#!/usr/bin/env python3
"""
Level 3 - Advanced: CI/CD Pipeline Automation with Python
"""

import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from enum import Enum

class Status(Enum):
    """Pipeline stage status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

class PipelineStage:
    """Represents a CI/CD pipeline stage"""
    
    def __init__(self, name, command=None, timeout=300):
        self.name = name
        self.command = command
        self.timeout = timeout
        self.status = Status.PENDING
        self.start_time = None
        self.end_time = None
        self.output = []
    
    def run(self):
        """Execute the stage"""
        self.status = Status.RUNNING
        self.start_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"Stage: {self.name}")
        print(f"{'='*60}")
        
        try:
            if self.command:
                # Execute command
                result = subprocess.run(
                    self.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                self.output = result.stdout.split('\n')
                
                if result.returncode == 0:
                    self.status = Status.SUCCESS
                    print(f"✓ {self.name} completed successfully")
                else:
                    self.status = Status.FAILED
                    print(f"✗ {self.name} failed")
                    if result.stderr:
                        print(f"Error: {result.stderr}")
            else:
                # Simulate stage execution
                time.sleep(1)
                self.status = Status.SUCCESS
                print(f"✓ {self.name} completed")
            
        except subprocess.TimeoutExpired:
            self.status = Status.FAILED
            print(f"✗ {self.name} timed out")
        except Exception as e:
            self.status = Status.FAILED
            print(f"✗ {self.name} failed: {e}")
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"Duration: {duration:.2f}s")
        
        return self.status == Status.SUCCESS
    
    def to_dict(self):
        """Convert stage to dictionary"""
        return {
            'name': self.name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': (self.end_time - self.start_time).total_seconds() 
                       if self.start_time and self.end_time else 0
        }

class CICDPipeline:
    """CI/CD Pipeline orchestrator"""
    
    def __init__(self, name, version):
        self.name = name
        self.version = version
        self.stages = []
        self.start_time = None
        self.end_time = None
    
    def add_stage(self, stage):
        """Add a stage to the pipeline"""
        self.stages.append(stage)
    
    def run(self, fail_fast=True):
        """Execute all pipeline stages"""
        self.start_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"CI/CD Pipeline: {self.name}")
        print(f"Version: {self.version}")
        print(f"Start Time: {self.start_time}")
        print(f"{'='*60}")
        
        success = True
        
        for stage in self.stages:
            if not stage.run():
                success = False
                if fail_fast:
                    print("\n✗ Pipeline failed - stopping execution")
                    break
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        print(f"\n{'='*60}")
        if success:
            print("✓ Pipeline completed successfully!")
        else:
            print("✗ Pipeline failed")
        print(f"Total Duration: {duration:.2f}s")
        print(f"{'='*60}")
        
        return success
    
    def generate_report(self, output_file='pipeline_report.json'):
        """Generate pipeline execution report"""
        report = {
            'pipeline': self.name,
            'version': self.version,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': (self.end_time - self.start_time).total_seconds() 
                       if self.start_time and self.end_time else 0,
            'stages': [stage.to_dict() for stage in self.stages],
            'success': all(s.status == Status.SUCCESS for s in self.stages)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report generated: {output_file}")

def main():
    """Main pipeline execution"""
    
    # Create pipeline
    pipeline = CICDPipeline(
        name="My Application",
        version=datetime.now().strftime("%Y%m%d.%H%M%S")
    )
    
    # Add stages
    pipeline.add_stage(PipelineStage("Checkout", "echo 'Cloning repository...'"))
    pipeline.add_stage(PipelineStage("Build", "echo 'Building application...'"))
    pipeline.add_stage(PipelineStage("Unit Tests", "echo 'Running tests...'"))
    pipeline.add_stage(PipelineStage("Code Quality", "echo 'Analyzing code quality...'"))
    pipeline.add_stage(PipelineStage("Security Scan", "echo 'Scanning for vulnerabilities...'"))
    pipeline.add_stage(PipelineStage("Docker Build", "echo 'Building Docker image...'"))
    pipeline.add_stage(PipelineStage("Deploy to Staging", "echo 'Deploying to staging...'"))
    pipeline.add_stage(PipelineStage("Integration Tests", "echo 'Running integration tests...'"))
    pipeline.add_stage(PipelineStage("Health Check", "echo 'Checking application health...'"))
    
    # Run pipeline
    success = pipeline.run(fail_fast=True)
    
    # Generate report
    pipeline.generate_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
