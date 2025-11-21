from euriai.langchain import create_chat_model

def get_chat_model(api_key: str):
    """
    Creates and returns a chat model using the provided API key.

    Args:
        api_key (str): The API key for accessing the chat model.

    Returns:
        ChatModel: An instance of the chat model.

    Raises:
        Exception: Raises an exception if the model cannot be created.
    """
    try:
        return create_chat_model(api_key=api_key, model="gpt-4.1-nano", temperature=0.7)
    except Exception as e:
        raise Exception(f"Failed to create chat model: {str(e)}")

def ask_chat_model(chat_model, prompt: str):
    """
    Sends a prompt to the chat model and retrieves the response.

    Args:
        chat_model: The chat model instance to use.
        prompt (str): The prompt to send to the chat model.

    Returns:
        str: The content of the response from the chat model.

    Raises:
        Exception: Raises an exception if the model does not return a valid response.
    """
    try:
        response = chat_model.invoke(prompt)
        return response.content if response else "No response received."
    except Exception as e:
        raise Exception(f"Error during query to chat model: {str(e)}")
