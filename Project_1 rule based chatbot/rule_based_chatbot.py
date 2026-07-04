# ============================================================
#  Simple Rule-Based Chatbot
#  Decode Labs Internship — Project 1
#
#  Requirements covered:
#  - Handle greetings and exit commands
#  - Use if-else logic for responses
#  - Run in a continuous loop
#
#  Skills demonstrated:
#  - Control flow (if/elif/else)
#  - Decision-making logic
#  - Basic AI concepts (pattern matching, conversational rules)
# ============================================================


def get_response(user_input):
    """
    Takes the user's message and returns a response
    using if-else decision logic.

    Returns the special value "EXIT" if the user wants to quit.
    """
    # Clean the input: lowercase + remove extra spaces
    # This makes "Hello", "HELLO", and " hello " all match the same rule
    message = user_input.lower().strip()

    # ----- Greetings -----
    if message in ["hello", "hi", "hey", "good morning", "good evening"]:
        return "Hello! How can I help you today?"

    # ----- How are you -----
    elif message in ["how are you", "how are you doing", "how's it going"]:
        return "I'm doing great, thank you! How about you?"

    # ----- Bot identity -----
    elif message in ["what is your name", "what's your name", "who are you"]:
        return "I'm a simple rule-based chatbot built for Decode Labs!"

    # ----- Capabilities -----
    elif message in ["what can you do", "help"]:
        return "I can greet you, answer simple questions, and chat a little. Try saying hello!"

    # ----- Thanks -----
    elif message in ["thank you", "thanks", "thx"]:
        return "You're welcome!"

    # ----- Bot's feelings -----
    elif message in ["are you a robot", "are you human"]:
        return "I'm a chatbot, made of code and logic, not flesh and blood!"

    # ----- Exit commands -----
    elif message in ["bye", "goodbye", "exit", "quit", "stop"]:
        return "EXIT"

    # ----- Fallback for anything unrecognized -----
    else:
        return "Sorry, I don't understand that. Try saying 'hello' or 'help'."


def run_chatbot():
    """
    Main chatbot loop.
    Keeps running continuously until the user types an exit command.
    """
    print("=" * 50)
    print("  Simple Rule-Based Chatbot")
    print("  Type 'hello' to start, or 'exit' to quit.")
    print("=" * 50)

    # Continuous loop — keeps the chatbot running
    while True:

        # Step 1: Get input from the user
        user_input = input("\nYou: ")

        # Step 2: Handle empty input
        if not user_input.strip():
            print("Bot: Please type something!")
            continue

        # Step 3: Get the bot's response using if-else logic
        response = get_response(user_input)

        # Step 4: Check if the user wants to exit
        if response == "EXIT":
            print("Bot: Goodbye! Have a great day!")
            break  # exits the while loop, ending the program

        # Step 5: Print the bot's response
        print(f"Bot: {response}")


# ============================================================
#  Entry point — runs when this file is executed directly
# ============================================================
if __name__ == "__main__":
    run_chatbot()
