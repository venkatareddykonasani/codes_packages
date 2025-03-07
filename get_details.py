def get_user_details():
    """
    Function to take user details as input.
    Returns:
        dict: A dictionary containing user details (name, MSRIT ID, email, mobile number)
    """
    name = input("Enter your name: ")
    msrit_id = input("Enter your MSRIT ID number: ")
    email = input("Enter your email: ")
    mobile = input("Enter your mobile number: ")
    
    return {
        "name": name,
        "msrit_id": msrit_id,
        "email": email,
        "mobile": mobile
    }

def main():
    """
    Main function to execute the user details input function.
    """
    user_details = get_user_details()
    print("\nUser Details:")
    for key, value in user_details.items():
        #print(f"{key.capitalize()}: {value}")

if __name__ == "__main__":
    main()
