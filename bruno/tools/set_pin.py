import sys
from bruno.auth.pin import set_pin

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m bruno.tools.set_pin <user_id> <pin>")
        raise SystemExit(2)
    user_id = sys.argv[1]
    pin = sys.argv[2]
    set_pin("data/users", user_id, pin)
    print(f"OK: PIN set for {user_id}")

if __name__ == "__main__":
    main()
