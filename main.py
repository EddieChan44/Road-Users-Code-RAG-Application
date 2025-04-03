# main.py
import sys

def main():
    while True:
        print("\n=== RAG LLM Application ===")
        print("1. Train / Create Vector Database")
        print("2. Chat with RAG LLM")
        print("3. Run Tests")
        print("q. Quit")
        
        selection = input("Enter your choice: ").strip()
        
        if selection == "1":
            print("\nStarting training process...")
            try:
                import train
                train.run_training()
            except Exception as e:
                print(f"An error occurred during training: {e}")
        elif selection == "2":
            print("\nStarting chat session...")
            try:
                import chat
                chat.run_chat()
            except Exception as e:
                print(f"An error occurred during chat: {e}")
        elif selection == "3":
            print("\nStarting testing process...")
            try:
                import test
                test.run_tests()
            except Exception as e:
                print(f"An error occurred during testing: {e}")
        elif selection.lower() in ("q", "quit", "exit"):
            print("\nExiting application. Goodbye!")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()