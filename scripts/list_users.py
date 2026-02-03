"""
Script to list all authorized users in the database
"""

from src.core.database_manager import DatabaseManager


def main():
    """List all users in the database"""
    db_manager = DatabaseManager()
    
    users = db_manager.get_all_users()
    
    if not users:
        print("\n[INFO] No authorized users in database")
        print("[INFO] Use 'python enroll_user.py --name <name>' to add users")
    else:
        print(f"\n[INFO] Found {len(users)} authorized user(s):\n")
        for i, user in enumerate(users, 1):
            embeddings = db_manager.get_user_embeddings(user)
            print(f"{i}. {user} ({len(embeddings)} sample(s))")
    
    print()


if __name__ == '__main__':
    main()
