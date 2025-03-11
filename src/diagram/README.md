# Architecture Diagram by Python Code

This directory contains an example of creating architecture diagrams using Python code.

## Getting Started

To get started with creating architecture diagrams, you can refer to the following link:
[Diagrams Documentation - Getting Started Examples](https://diagrams.mingrammer.com/docs/getting-started/examples)

## Example

Here is a basic example of how to create a simple architecture diagram using the `diagrams` library in Python:

1. Import the necessary modules from the `diagrams` library.
2. Define the components and their relationships within a `Diagram` context.

Example Code:
- Import the `Diagram` class and components:
```python
  from diagrams import Diagram
  from diagrams.aws.compute import EC2
  from diagrams.aws.network import ELB
  from diagrams.aws.database import RDS
```
- Create the diagram:
```python
  with Diagram("Simple Architecture Diagram", show=False):
    ELB("load balancer") >> EC2("web server") >> RDS("database")
```

## Installation

To install the `diagrams` library, you can use the following command:
- `pip install diagrams`

## Usage

1. Create your Python script with the architecture diagram code.
2. Run the script to generate the diagram image.

For more detailed examples and usage, please visit the [official documentation](https://diagrams.mingrammer.com/docs/getting-started/examples).

Feel free to contribute by adding more examples or improving the existing ones.
