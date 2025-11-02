import ast

def parse_relation_string(s: str) -> dict:
    """
    Parses a string representation of a dictionary with tuple keys and list values
    into an actual Python dictionary.

    Example input:
    '{("ANSWER", "A"): ["phenotype present"], ("ANSWER", "B"): ["phenotype present"]}'
    """
    try:
        parsed_dict = ast.literal_eval(s)
        if isinstance(parsed_dict, dict):
            return parsed_dict
        else:
            raise ValueError("Input string is not a dictionary.")
    except Exception as e:
        raise ValueError(f"Invalid format: {e}")

