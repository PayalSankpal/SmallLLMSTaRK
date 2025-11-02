import json
from typing import Dict, Any, List, Union


def parse_entity_response(response: str) -> Dict[str, Any]:
    """
    Parse the entity extraction JSON response and return it as a dictionary.
    
    Args:
        response: JSON string containing extracted entities
        
    Returns:
        Dictionary containing parsed entity information
        
    Raises:
        ValueError: If the response is not valid JSON or doesn't match expected format
    """
    try:
        # Parse the JSON string
        entities = json.loads(response)
        
        # Validate the structure
        if not isinstance(entities, dict):
            raise ValueError("Response must be a JSON object (dictionary)")
        
        # Validate each entity
        for entity_key, entity_data in entities.items():
            validate_entity(entity_key, entity_data)
        
        return entities
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")


def validate_entity(entity_key: str, entity_data: Dict[str, Any]) -> None:
    """
    Validate that an entity has the correct structure.
    
    Args:
        entity_key: The entity identifier (e.g., "A", "B", "ANSWER")
        entity_data: The entity data dictionary
        
    Raises:
        ValueError: If the entity structure is invalid
    """
    required_fields = ["type", "lexical", "semantic", "constant"]
    
    # Check all required fields are present
    for field in required_fields:
        if field not in entity_data:
            raise ValueError(f"Entity '{entity_key}' missing required field: '{field}'")
    
    # Validate 'type' field
    if not isinstance(entity_data["type"], list):
        raise ValueError(f"Entity '{entity_key}': 'type' must be a list")
    
    if not all(isinstance(t, str) for t in entity_data["type"]):
        raise ValueError(f"Entity '{entity_key}': all 'type' values must be strings")
    
    # Validate 'lexical' field
    if not isinstance(entity_data["lexical"], dict):
        raise ValueError(f"Entity '{entity_key}': 'lexical' must be a dictionary")
    
    # Validate 'semantic' field
    if not isinstance(entity_data["semantic"], list):
        raise ValueError(f"Entity '{entity_key}': 'semantic' must be a list")
    
    if not all(isinstance(s, str) for s in entity_data["semantic"]):
        raise ValueError(f"Entity '{entity_key}': all 'semantic' values must be strings")
    
    # Validate 'constant' field
    if not isinstance(entity_data["constant"], bool):
        raise ValueError(f"Entity '{entity_key}': 'constant' must be a boolean")


def extract_entity_info(entities: Dict[str, Any], entity_key: str) -> Dict[str, Any]:
    """
    Extract information for a specific entity.
    
    Args:
        entities: Dictionary of all entities
        entity_key: The key of the entity to extract (e.g., "A", "ANSWER")
        
    Returns:
        Dictionary containing the entity's information
        
    Raises:
        KeyError: If the entity_key doesn't exist
    """
    if entity_key not in entities:
        raise KeyError(f"Entity '{entity_key}' not found in response")
    
    return entities[entity_key]


def get_answer_entity(entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the ANSWER entity from the entities dictionary.
    
    Args:
        entities: Dictionary of all entities
        
    Returns:
        Dictionary containing the ANSWER entity's information
        
    Raises:
        KeyError: If ANSWER entity doesn't exist
    """
    return extract_entity_info(entities, "ANSWER")


def list_entity_keys(entities: Dict[str, Any]) -> List[str]:
    """
    Get a list of all entity keys in the response.
    
    Args:
        entities: Dictionary of all entities
        
    Returns:
        List of entity keys
    """
    return list(entities.keys())


def get_entities_by_type(entities: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
    """
    Get all entities of a specific type.
    
    Args:
        entities: Dictionary of all entities
        entity_type: The type to filter by (e.g., "drug", "disease", "gene/protein")
        
    Returns:
        Dictionary of entities that have the specified type
    """
    filtered_entities = {}
    
    for key, entity_data in entities.items():
        if entity_type in entity_data["type"]:
            filtered_entities[key] = entity_data
    
    return filtered_entities


# Example usage
if __name__ == "__main__":
    # Example response JSON
    response_json = """
    {
        "A": {
            "type": ["gene/protein"],
            "lexical": {"name": "CYP3A4"},
            "semantic": [],
            "constant": true
        },
        "B": {
            "type": ["disease"],
            "lexical": {"name": "strongyloidiasis"},
            "semantic": [],
            "constant": true
        },
        "ANSWER": {
            "type": ["drug"],
            "lexical": {},
            "semantic": ["used to treat strongyloidiasis"],
            "constant": false
        }
    }
    """
    
    # Parse the response
    entities = parse_entity_response(response_json)
    
    # Print all entities
    print("All entities:", entities)
    print()
    
    # Get the ANSWER entity
    answer = get_answer_entity(entities)
    print("ANSWER entity:", answer)
    print()
    
    # List all entity keys
    keys = list_entity_keys(entities)
    print("Entity keys:", keys)
    print()
    
    # Get all drug entities
    drugs = get_entities_by_type(entities, "drug")
    print("Drug entities:", drugs)
    print()
    
    # Get specific entity info
    entity_a = extract_entity_info(entities, "A")
    print("Entity A:", entity_a)
    print("Entity A name:", entity_a["lexical"].get("name", "No name specified"))