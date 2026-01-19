#!/bin/bash
# Level 2 - Intermediate: File Operations and Backup Script

# Configuration
SOURCE_DIR="./data"
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="backup_${TIMESTAMP}.tar.gz"

# Create directories if they don't exist
mkdir -p "$SOURCE_DIR"
mkdir -p "$BACKUP_DIR"

# Create sample data
echo "Creating sample data..."
echo "This is a test file" > "$SOURCE_DIR/test.txt"
echo "Another file for backup" > "$SOURCE_DIR/file2.txt"

# Function to create backup
create_backup() {
    local source=$1
    local dest=$2
    local backup_file=$3
    
    echo "Creating backup..."
    if tar -czf "$dest/$backup_file" -C "$source" .; then
        echo "✓ Backup created successfully: $dest/$backup_file"
        return 0
    else
        echo "✗ Backup failed"
        return 1
    fi
}

# Function to list backups
list_backups() {
    echo "Available backups:"
    ls -lh "$BACKUP_DIR"
}

# Function to cleanup old backups (keep last 5)
cleanup_old_backups() {
    local backup_dir=$1
    local keep_count=5
    
    echo "Cleaning up old backups (keeping last $keep_count)..."
    
    cd "$backup_dir" || return 1
    backup_count=$(ls -1 backup_*.tar.gz 2>/dev/null | wc -l)
    
    if [ "$backup_count" -gt "$keep_count" ]; then
        ls -t backup_*.tar.gz | tail -n +$((keep_count + 1)) | xargs rm -f
        echo "✓ Cleanup completed"
    else
        echo "No cleanup needed"
    fi
}

# Main execution
echo "=== Backup Script ==="
create_backup "$SOURCE_DIR" "$BACKUP_DIR" "$BACKUP_NAME"
echo ""
list_backups
echo ""
cleanup_old_backups "$BACKUP_DIR"

echo ""
echo "Backup script completed!"
