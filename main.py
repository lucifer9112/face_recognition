# main.py

from face_system import FaceSystem

def main():
    system = FaceSystem()

    while True:
        print("\n" + "=" * 50)
        print(" LBPH LIVE FACE RECOGNITION")
        print("=" * 50)
        print("1. Add new person")
        print("2. Train LBPH model")
        print("3. Start live recognition")
        print("4. Show stats")
        print("5. Exit")
        print("=" * 50)

        choice = input("Choice (1-5): ").strip()

        if choice == "1":
            name = input("Name: ").strip()
            if not name:
                print("Name empty.")
                continue
            try:
                samples = int(input("Samples [default 50]: ") or "50")
            except ValueError:
                print("❌ Invalid number. Using default 50.")
                samples = 50
            system.add_person(name, samples)

        elif choice == "2":
            system.train_lbph()

        elif choice == "3":
            system.start()

        elif choice == "4":
            print(system.stats)

        elif choice == "5":
            print("Bye.")
            break

        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
