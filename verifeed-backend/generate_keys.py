# generate_keys.py
import secrets

print("=" * 70)
print("VeriFeed Security Keys Generator")
print("=" * 70)
print("\nCopy these values to .env file:\n")
print("-" * 70)
print(f"JWT_SECRET_KEY={secrets.token_hex(32)}")
print(f"API_KEY={secrets.token_urlsafe(32)}")
print(f"ADMIN_API_KEY={secrets.token_urlsafe(32)}")
print("-" * 70)

print("\n⚠️  IMPORTANT:")
print("1. Save these keys securely")
print("2. Never commit them to GitHub")
print("3. Use the same API_KEY in your extension")
print("=" * 70)