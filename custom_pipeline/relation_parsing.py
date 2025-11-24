import ast
import re

def parse_relation_string(response_string):
    """Parse LLM response containing relation tuples as string keys"""
    
    # Remove markdown code blocks if present
    response_string = response_string.strip()
    if response_string.startswith('```'):
        response_string = re.sub(r'^```(?:json)?\s*', '', response_string)
        response_string = re.sub(r'```\s*$', '', response_string)
    
    response_string = response_string.strip()
    
    if not response_string or response_string == '{}':
        return {}
    
    try:
        # Try using ast.literal_eval instead of json.loads
        # This can parse Python dict syntax with tuple keys
        parsed = ast.literal_eval(response_string)
        
        # If keys are already tuples, we're good
        result = {}
        for key, value in parsed.items():
            if isinstance(key, tuple):
                result[key] = value
            elif isinstance(key, str) and key.startswith('(') and key.endswith(')'):
                # Handle string representation of tuples
                entities = re.findall(r'"([^"]+)"', key)
                if len(entities) == 2:
                    tuple_key = tuple(entities)
                    result[tuple_key] = value
                else:
                    result[key] = value
            else:
                result[key] = value
        
        return result
        
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse relation string: {e}\nOriginal: {response_string}")