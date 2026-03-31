from model import train_model, predict

print("Training model...")
model, vectorizer = train_model()

while True:
    print("\n===== Spam Classifier =====")
    print("1. Check Message")
    print("2. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        text = input("Enter message: ")

        if text.strip() == "":
            print("Empty input not allowed")
            continue

        result = predict(text, model, vectorizer)
        print("Result:", result)

    elif choice == "2":
        print("Exiting...")
        break

    else:
        print("Invalid choice")