[← Back to Python Study Repository](../../README.md)

# Shell Automation Learning - Zero to Expert

A comprehensive learning path for mastering shell scripting, automation, DevOps, and CI/CD across multiple platforms (Bash, PowerShell, Batch, and Python).

## Table of Contents

- [Introduction](#introduction)
- [Learning Path](#learning-path)
- [Technologies Covered](#technologies-covered)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [Learning Levels](#learning-levels)
- [Prerequisites](#prerequisites)
- [Contributing](#contributing)

## Introduction

This project provides a structured learning path from zero to expert level in shell automation, DevOps practices, and CI/CD pipelines. Whether you're a beginner or looking to advance your automation skills, this repository offers hands-on examples and real-world scenarios across multiple scripting languages.

## Learning Path

The learning path is divided into 6 progressive levels:

1. **[Level 0 - Beginner](level-0-beginner/README.md)**: Basic syntax, commands, and simple scripts
2. **[Level 1 - Basic](level-1-basic/README.md)**: Variables, conditionals, loops, and functions
3. **[Level 2 - Intermediate](level-2-intermediate/README.md)**: File operations, error handling, and automation tasks
4. **[Level 3 - Advanced](level-3-advanced/README.md)**: CI/CD pipelines, Docker, configuration management
5. **[Level 4 - Expert](level-4-expert/README.md)**: Advanced DevOps patterns, orchestration, multi-platform automation
6. **[Level 5 - Master](level-5-master/README.md)**: Production-ready systems, security, monitoring, and best practices

## Technologies Covered

- **Bash**: Linux/Unix shell scripting for automation
- **PowerShell**: Windows automation and cross-platform scripting
- **Batch**: Windows batch file scripting
- **Python**: Automation scripts for DevOps tasks

### Focus Areas

- Shell scripting fundamentals
- Automation and task scheduling
- DevOps practices and workflows
- CI/CD pipeline creation and management
- Docker and containerization
- Configuration management
- Infrastructure as Code (IaC)
- Monitoring and logging
- Security best practices

## Directory Structure

```
learning-shell-automation/
├── README.md
├── level-0-beginner/          # Introduction to shell scripting
│   ├── bash/
│   ├── powershell/
│   ├── batch/
│   └── python/
├── level-1-basic/             # Basic scripting concepts
│   ├── bash/
│   ├── powershell/
│   ├── batch/
│   └── python/
├── level-2-intermediate/      # Automation and file operations
│   ├── bash/
│   ├── powershell/
│   ├── batch/
│   └── python/
├── level-3-advanced/          # CI/CD and DevOps practices
│   ├── bash/
│   ├── powershell/
│   ├── batch/
│   └── python/
├── level-4-expert/            # Advanced patterns and orchestration
│   ├── bash/
│   ├── powershell/
│   ├── batch/
│   └── python/
└── level-5-master/            # Production-ready systems
    ├── bash/
    ├── powershell/
    ├── batch/
    └── python/
```

## Getting Started

### For Linux/macOS Users (Bash)

1. Navigate to the bash directory:
   ```bash
   cd src/learning-shell-automation/level-0-beginner/bash
   ```

2. Make scripts executable:
   ```bash
   chmod +x *.sh
   ```

3. Run a script:
   ```bash
   ./hello_world.sh
   ```

### For Windows Users (PowerShell)

1. Navigate to the PowerShell directory:
   ```powershell
   cd src\learning-shell-automation\level-0-beginner\powershell
   ```

2. Set execution policy (if needed):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. Run a script:
   ```powershell
   .\hello_world.ps1
   ```

### For Windows Users (Batch)

1. Navigate to the batch directory:
   ```cmd
   cd src\learning-shell-automation\level-0-beginner\batch
   ```

2. Run a script:
   ```cmd
   hello_world.bat
   ```

### For Python Users

1. Navigate to the python directory:
   ```bash
   cd src/learning-shell-automation/level-0-beginner/python
   ```

2. Run a script:
   ```bash
   python hello_automation.py
   ```

## Learning Levels

### [Level 0 - Beginner](level-0-beginner/README.md)
- Hello World scripts
- Basic command execution
- Simple output and input
- Running your first automation

### [Level 1 - Basic](level-1-basic/README.md)
- Variables and data types
- Conditionals (if/else)
- Loops (for, while)
- Functions and parameters
- Basic string manipulation

### [Level 2 - Intermediate](level-2-intermediate/README.md)
- File operations (read, write, copy, move)
- Error handling and debugging
- Regular expressions
- Process management
- Scheduled tasks and cron jobs
- Environment variables

### [Level 3 - Advanced](level-3-advanced/README.md)
- CI/CD pipeline scripts
- Docker integration
- Configuration management (Ansible, Terraform basics)
- API interactions
- Database operations
- Log parsing and analysis

### [Level 4 - Expert](level-4-expert/README.md)
- Multi-platform automation
- Advanced orchestration
- Kubernetes automation
- Infrastructure as Code
- Custom CI/CD tools
- Performance optimization

### [Level 5 - Master](level-5-master/README.md)
- Production-ready deployment systems
- Advanced security practices
- Monitoring and alerting systems
- Disaster recovery automation
- High availability setups
- Best practices and design patterns

## Prerequisites

### System Requirements

- **For Bash**: Linux, macOS, or WSL (Windows Subsystem for Linux)
- **For PowerShell**: Windows 10+, or PowerShell Core (cross-platform)
- **For Batch**: Windows
- **For Python**: Python 3.6+

### Recommended Tools

- Git for version control
- Docker for containerization examples
- A code editor (VS Code, Vim, Sublime Text, etc.)
- Terminal/Command Prompt with administrator privileges

### Optional but Recommended

- GitHub account for CI/CD examples
- Docker Desktop
- Cloud provider account (AWS, Azure, or GCP) for cloud automation examples

## Best Practices

1. **Always test scripts in a safe environment** before running in production
2. **Use version control** for all your scripts
3. **Add comments** to explain complex logic
4. **Handle errors gracefully** with proper error checking
5. **Use meaningful variable names**
6. **Follow security best practices** (never hardcode credentials)
7. **Make scripts idempotent** when possible
8. **Log important operations** for debugging and auditing

## Contributing

Contributions are welcome! If you have improvements, bug fixes, or new examples to add:

1. Fork the repository
2. Create a feature branch
3. Add your scripts with proper documentation
4. Submit a pull request

Please ensure your scripts follow the project structure and include:
- Clear comments explaining the code
- README files for complex examples
- Error handling where appropriate
- Examples of expected output

## Resources

### Official Documentation
- [Bash Manual](https://www.gnu.org/software/bash/manual/)
- [PowerShell Documentation](https://docs.microsoft.com/en-us/powershell/)
- [Python Documentation](https://docs.python.org/3/)

### Learning Resources
- [Shell Scripting Tutorial](https://www.shellscript.sh/)
- [PowerShell Tutorial](https://learn.microsoft.com/en-us/training/paths/powershell/)
- [DevOps Practices](https://www.atlassian.com/devops)
- [CI/CD Best Practices](https://www.jenkins.io/doc/book/pipeline/)

### Community
- [Stack Overflow - Bash](https://stackoverflow.com/questions/tagged/bash)
- [Stack Overflow - PowerShell](https://stackoverflow.com/questions/tagged/powershell)
- [DevOps Subreddit](https://www.reddit.com/r/devops/)

## License

This project is part of the python-study repository and is licensed under the MIT License. See the main [LICENSE](../../LICENSE) file for details.

## Feedback

If you have questions, suggestions, or feedback, please open an issue in the main repository.

---

**Happy Learning! 🚀**
