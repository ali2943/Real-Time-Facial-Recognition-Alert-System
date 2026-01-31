"""
Script to remove a user from the database
"""

import argparse
from database_manager import DatabaseManager


def main():
    """Remove a user from the database"""
    parser = argparse.ArgumentParser(description='Remove user from database')
    parser.add_argument('--name', type=str, required=True, help='Name of user to remove')
    
    args = parser.parse_args()
    
    db_manager = DatabaseManager()
    
    # Check if user exists
    if args.name in db_manager.get_all_users():
        response = input(f"Are you sure you want to remove '{args.name}'? (yes/no): ")
        if response.lower() == 'yes':
            db_manager.remove_user(args.name)
            print(f"[SUCCESS] User '{args.name}' removed successfully")
        else:
            print("[INFO] Operation cancelled")
    else:
        print(f"[ERROR] User '{args.name}' not found in database")


if __name__ == '__main__':
    main()
