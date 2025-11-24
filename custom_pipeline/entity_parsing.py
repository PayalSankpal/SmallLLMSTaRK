import re
import json

def parse_entity_response(response_string):
    """Parse LLM response containing entity information"""
    
    # Remove markdown code blocks if present
    response_string = response_string.strip()
    if response_string.startswith('```'):
        response_string = re.sub(r'^```(?:json)?\s*', '', response_string)
        response_string = re.sub(r'```\s*$', '', response_string)
    
    response_string = response_string.strip()
    
    if not response_string:
        raise ValueError("Empty response after cleaning")
    
    try:
        # Parse the JSON
        entities = json.loads(response_string)
        
        # Validate structure
        if not isinstance(entities, dict):
            raise ValueError("Response is not a dictionary")
        
        # Validate each entity has required fields
        for entity_key, entity_data in entities.items():
            if not isinstance(entity_data, dict):
                raise ValueError(f"Entity {entity_key} is not a dictionary")
            
            required_fields = ['type', 'lexical', 'semantic', 'constant']
            for field in required_fields:
                if field not in entity_data:
                    raise ValueError(f"Entity {entity_key} missing required field: {field}")
        
        return entities
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
