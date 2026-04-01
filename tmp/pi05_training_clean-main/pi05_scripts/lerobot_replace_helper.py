#!/usr/bin/env python3
"""
Helper script to replace files in the installed lerobot library with custom versions.

This script:
1. Backs up original lerobot files to a specified directory
2. Replaces them with custom versions from a replacement directory
3. Restores original files from backup
4. Prints a summary of all operations performed
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def find_lerobot_install_path():
    """Find the installation path of the lerobot package."""
    try:
        import lerobot
        lerobot_path = Path(lerobot.__file__).parent
        return lerobot_path
    except ImportError:
        print("Error: lerobot package not found. Please install it first.")
        sys.exit(1)


def get_all_files(directory):
    """Recursively get all files in a directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(Path(root) / filename)
    return files


def backup_and_replace(replace_dir, backup_dir, lerobot_root):
    """
    Backup original files and replace them with new versions.

    Args:
        replace_dir: Directory containing replacement files
        backup_dir: Directory where original files will be backed up
        lerobot_root: Root directory of installed lerobot package

    Returns:
        List of tuples (relative_path, operation_status)
    """
    replace_dir = Path(replace_dir).resolve()
    backup_dir = Path(backup_dir).resolve()
    lerobot_root = Path(lerobot_root).resolve()

    if not replace_dir.exists():
        print(f"Error: Replacement directory does not exist: {replace_dir}")
        sys.exit(1)

    # Create backup directory if it doesn't exist
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Get all files in replacement directory
    replacement_files = get_all_files(replace_dir)

    operations = []

    for repl_file in replacement_files:
        # Get relative path from replacement directory
        rel_path = repl_file.relative_to(replace_dir)

        # Determine paths
        original_file = lerobot_root / rel_path
        backup_file = backup_dir / rel_path

        try:
            # Check if original file exists
            if not original_file.exists():
                operations.append((str(rel_path), "SKIPPED", f"Original file not found: {original_file}"))
                continue

            # Create backup directory structure if needed
            backup_file.parent.mkdir(parents=True, exist_ok=True)

            # Backup original file (only if not already backed up)
            if not backup_file.exists():
                shutil.copy2(original_file, backup_file)
                backup_status = "BACKED UP"
            else:
                backup_status = "ALREADY BACKED UP"

            # Replace with new file
            shutil.copy2(repl_file, original_file)

            operations.append((str(rel_path), "SUCCESS", backup_status))

        except Exception as e:
            operations.append((str(rel_path), "FAILED", str(e)))

    return operations


def restore_from_backup(backup_dir, lerobot_root):
    """
    Restore original files from backup directory to lerobot installation.

    Args:
        backup_dir: Directory containing backed up files
        lerobot_root: Root directory of installed lerobot package

    Returns:
        List of tuples (relative_path, operation_status, detail)
    """
    backup_dir = Path(backup_dir).resolve()
    lerobot_root = Path(lerobot_root).resolve()

    if not backup_dir.exists():
        print(f"Error: Backup directory does not exist: {backup_dir}")
        sys.exit(1)

    # Get all files in backup directory
    backup_files = get_all_files(backup_dir)

    if not backup_files:
        print(f"Warning: No files found in backup directory: {backup_dir}")
        return []

    operations = []

    for backup_file in backup_files:
        # Get relative path from backup directory
        rel_path = backup_file.relative_to(backup_dir)

        # Determine target path in lerobot installation
        target_file = lerobot_root / rel_path

        try:
            # Check if target location exists
            if not target_file.parent.exists():
                operations.append((str(rel_path), "SKIPPED", f"Target directory not found: {target_file.parent}"))
                continue

            # Restore file from backup
            shutil.copy2(backup_file, target_file)
            operations.append((str(rel_path), "SUCCESS", "Restored from backup"))

        except Exception as e:
            operations.append((str(rel_path), "FAILED", str(e)))

    return operations


def print_summary(operations, replace_dir, backup_dir, lerobot_root):
    """Print a summary of all operations performed."""
    print("\n" + "=" * 80)
    print("LEROBOT REPLACEMENT SUMMARY")
    print("=" * 80)
    print(f"Lerobot install path: {lerobot_root}")
    print(f"Replacement source:   {replace_dir}")
    print(f"Backup location:      {backup_dir}")
    print("=" * 80)

    success_count = sum(1 for _, status, _ in operations if status == "SUCCESS")
    failed_count = sum(1 for _, status, _ in operations if status == "FAILED")
    skipped_count = sum(1 for _, status, _ in operations if status == "SKIPPED")

    print(f"\nTotal files processed: {len(operations)}")
    print(f"  ✓ Successfully replaced: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  ⊘ Skipped: {skipped_count}")

    if operations:
        print("\nDetailed operations:")
        print("-" * 80)

        for rel_path, status, detail in operations:
            if status == "SUCCESS":
                print(f"✓ {rel_path}")
                print(f"  └─ {detail}")
            elif status == "FAILED":
                print(f"✗ {rel_path}")
                print(f"  └─ ERROR: {detail}")
            elif status == "SKIPPED":
                print(f"⊘ {rel_path}")
                print(f"  └─ {detail}")

    print("=" * 80)

    if success_count > 0:
        print("\n⚠️  Note: Changes have been made to your installed lerobot package.")
        print("   These changes will persist until you reinstall or restore from backup.")

    return success_count, failed_count, skipped_count


def print_restore_summary(operations, backup_dir, lerobot_root):
    """Print a summary of restore operations performed."""
    print("\n" + "=" * 80)
    print("LEROBOT RESTORE SUMMARY")
    print("=" * 80)
    print(f"Lerobot install path: {lerobot_root}")
    print(f"Backup source:        {backup_dir}")
    print("=" * 80)

    success_count = sum(1 for _, status, _ in operations if status == "SUCCESS")
    failed_count = sum(1 for _, status, _ in operations if status == "FAILED")
    skipped_count = sum(1 for _, status, _ in operations if status == "SKIPPED")

    print(f"\nTotal files processed: {len(operations)}")
    print(f"  ✓ Successfully restored: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  ⊘ Skipped: {skipped_count}")

    if operations:
        print("\nDetailed operations:")
        print("-" * 80)

        for rel_path, status, detail in operations:
            if status == "SUCCESS":
                print(f"✓ {rel_path}")
                print(f"  └─ {detail}")
            elif status == "FAILED":
                print(f"✗ {rel_path}")
                print(f"  └─ ERROR: {detail}")
            elif status == "SKIPPED":
                print(f"⊘ {rel_path}")
                print(f"  └─ {detail}")

    print("=" * 80)

    if success_count > 0:
        print("\n✓ Original files have been restored to your lerobot installation.")

    return success_count, failed_count, skipped_count


def main():
    parser = argparse.ArgumentParser(
        description="Replace or restore files in installed lerobot library",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)

    # Replace subcommand
    replace_parser = subparsers.add_parser(
        "replace",
        help="Replace lerobot files with custom versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replace files from ./my_modifications and backup to ./lerobot_backup
  python lerobot_replace_helper.py replace ./my_modifications ./lerobot_backup

  # Use custom paths
  python lerobot_replace_helper.py replace /path/to/lerobot_replace /path/to/lerobot_orig
        """
    )
    replace_parser.add_argument(
        "replace_dir",
        help="Directory containing replacement files (with same structure as lerobot)"
    )
    replace_parser.add_argument(
        "backup_dir",
        help="Directory where original files will be backed up"
    )

    # Restore subcommand
    restore_parser = subparsers.add_parser(
        "restore",
        help="Restore original lerobot files from backup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Restore files from backup directory
  python lerobot_replace_helper.py restore ./lerobot_backup

  # Restore from custom backup location
  python lerobot_replace_helper.py restore /path/to/lerobot_orig
        """
    )
    restore_parser.add_argument(
        "backup_dir",
        help="Directory containing backed up original files"
    )

    args = parser.parse_args()

    # Find lerobot installation
    print("Finding lerobot installation...")
    lerobot_root = find_lerobot_install_path()
    print(f"Found lerobot at: {lerobot_root}")

    if args.command == "replace":
        # Perform backup and replacement
        print(f"\nProcessing files from: {args.replace_dir}")
        operations = backup_and_replace(args.replace_dir, args.backup_dir, lerobot_root)

        # Print summary
        success, failed, skipped = print_summary(
            operations,
            args.replace_dir,
            args.backup_dir,
            lerobot_root
        )

        # Exit with appropriate code
        if failed > 0:
            sys.exit(1)
        elif success == 0:
            print("\n⚠️  Warning: No files were replaced!")
            sys.exit(0)
        else:
            sys.exit(0)

    elif args.command == "restore":
        # Perform restore
        print(f"\nRestoring files from backup: {args.backup_dir}")
        operations = restore_from_backup(args.backup_dir, lerobot_root)

        # Print summary
        success, failed, skipped = print_restore_summary(
            operations,
            args.backup_dir,
            lerobot_root
        )

        # Exit with appropriate code
        if failed > 0:
            sys.exit(1)
        elif success == 0:
            print("\n⚠️  Warning: No files were restored!")
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()