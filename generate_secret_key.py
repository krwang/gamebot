import secrets

# Generate a 32-byte (256-bit) random key
secret_key = secrets.token_hex(32)
print(f"Generated Flask Secret Key: {secret_key}") 