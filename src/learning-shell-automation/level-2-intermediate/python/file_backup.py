#!/usr/bin/env python3
"""
Level 2 - Intermediate: File Backup Automation with Python
"""

import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

class BackupManager:
    """Manages file backups with rotation"""
    
    def __init__(self, source_dir, backup_dir, max_backups=5):
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        
        # Create directories if they don't exist
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self):
        """Create a compressed backup of the source directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}.tar.gz"
        backup_path = self.backup_dir / backup_name
        
        print(f"Creating backup: {backup_name}")
        
        try:
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(self.source_dir, arcname=os.path.basename(self.source_dir))
            
            print(f"✓ Backup created: {backup_path}")
            print(f"  Size: {self._get_file_size(backup_path)}")
            return True
        except Exception as e:
            print(f"✗ Backup failed: {e}")
            return False
    
    def list_backups(self):
        """List all available backups"""
        backups = sorted(self.backup_dir.glob("backup_*.tar.gz"))
        
        print("\nAvailable backups:")
        if not backups:
            print("  No backups found")
            return []
        
        for backup in backups:
            size = self._get_file_size(backup)
            modified = datetime.fromtimestamp(backup.stat().st_mtime)
            print(f"  {backup.name:30} {size:>10} {modified}")
        
        return backups
    
    def cleanup_old_backups(self):
        """Remove old backups, keeping only the most recent ones"""
        backups = sorted(self.backup_dir.glob("backup_*.tar.gz"), 
                        key=lambda x: x.stat().st_mtime)
        
        if len(backups) <= self.max_backups:
            print(f"\nNo cleanup needed (keeping {self.max_backups} backups)")
            return
        
        print(f"\nCleaning up old backups (keeping last {self.max_backups})...")
        
        to_delete = backups[:-self.max_backups]
        for backup in to_delete:
            backup.unlink()
            print(f"  Deleted: {backup.name}")
        
        print(f"✓ Cleanup completed ({len(to_delete)} files removed)")
    
    def restore_backup(self, backup_name, restore_dir):
        """Restore a specific backup"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            print(f"✗ Backup not found: {backup_name}")
            return False
        
        restore_path = Path(restore_dir)
        restore_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Restoring backup: {backup_name}")
        
        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(restore_path)
            
            print(f"✓ Backup restored to: {restore_path}")
            return True
        except Exception as e:
            print(f"✗ Restore failed: {e}")
            return False
    
    def _get_file_size(self, file_path):
        """Get human-readable file size"""
        size = file_path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"

def main():
    # Configuration
    source_dir = "./data"
    backup_dir = "./backups"
    
    # Create sample data
    print("Creating sample data...")
    Path(source_dir).mkdir(exist_ok=True)
    
    with open(f"{source_dir}/test.txt", "w") as f:
        f.write("This is a test file for backup\n")
    
    with open(f"{source_dir}/config.json", "w") as f:
        f.write('{"app": "backup_demo", "version": "1.0"}\n')
    
    print("✓ Sample data created\n")
    
    # Create backup manager
    manager = BackupManager(source_dir, backup_dir, max_backups=3)
    
    # Create backup
    print("=== Creating Backup ===")
    manager.create_backup()
    
    # List backups
    print("\n=== Listing Backups ===")
    manager.list_backups()
    
    # Cleanup old backups
    print("\n=== Cleanup Old Backups ===")
    manager.cleanup_old_backups()
    
    print("\n✓ Backup automation demo completed!")

if __name__ == "__main__":
    main()
