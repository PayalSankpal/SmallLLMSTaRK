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
        parsed = ast.literal_eval(response_string)
        
        result = {}
        for key, value in parsed.items():
            if isinstance(key, tuple):
                result[key] = value
            elif isinstance(key, str):
                # Clean the string key
                clean_key = key.strip()
                
                # Check if it looks like a tuple: starts with ( and ends with )
                if clean_key.startswith('(') and clean_key.endswith(')'):
                    try:
                        # Attempt 1: Try safe evaluation (works for "('A', 'B')")
                        potential_tuple = ast.literal_eval(clean_key)
                        if isinstance(potential_tuple, tuple):
                            result[potential_tuple] = value
                            continue
                    except (ValueError, SyntaxError):
                        pass

                    # Attempt 2: Regex for quoted strings inside parens (works for "('A', 'B')")
                    # NEW: handle cases where quotes might be mixed
                    entities_quoted = re.findall(r'[\'"]([^\'"]+)[\'"]', clean_key)
                    if len(entities_quoted) == 2:
                        result[tuple(entities_quoted)] = value
                        continue

                    # Attempt 3: Regex for unquoted strings inside parens (works for "(A, ANSWER)")
                    # NEW: Robust fallback for missing quotes
                    inner_content = clean_key[1:-1] # Remove ( and )
                    if ',' in inner_content:
                        # Split by comma and strip whitespace
                        parts = [p.strip() for p in inner_content.split(',', 1)]
                        if len(parts) == 2:
                            # Remove any lingering quotes just in case
                            clean_parts = [p.strip("'\"") for p in parts]
                            result[tuple(clean_parts)] = value
                            continue
                    
                    # If all parsing fails, keep original string key
                    result[key] = value
                else:
                    result[key] = value
            else:
                result[key] = value
        
        return result
        
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse relation string: {e}\nOriginal: {response_string}")