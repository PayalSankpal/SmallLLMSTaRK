def get_entity_extraction_prompt(query: str, dataset_name: str) -> str:

    base_prompt = """ENTITY EXTRACTION TASK

    OBJECTIVE: Extract entities from a natural language query and structure them in JSON format. One entity must be designated as "ANSWER", and others should use placeholder names (A, B, C, etc.).

    EXTRACTION RULES

    Rule 1: Entity Types
    Only extract entities mentioned in the query.For each entity, provide a list of possible types from this list: {entity_types_placeholder}. THE ENTITY TYPE SHOULD NOT BE ANY OTHER THAN THESE. An entity may have multiple types.
    If a general entity is mentioned in the query , extract it as an entity too and IF there is no semantic information for it, use the entire query as a semantic constraint for it, else use the semantic .
    DO NOT EVER assign an entity a possible type not in this list.

    Rule 2: Properties
    If an entity has specific properties mentioned in the query, include them. Properties must be from this list: title, name.

    Rule 3: Lexical Constraints
    Identify hard lexical constraints where field values must match exactly. Do NOT use general names. Use specific entity names only. for entities that do not have their name mentioned in the query, add the general information in the semantic part.

    Rule 4: Semantic Constraints
    Provide a general description of the entity. Include ALL relevant information from the query. Can include descriptions, short phrases, and contextual information. Format as a list of descriptive strings.

    Rule 5: Constant Field
    Boolean field indicating whether the entity is a specific constant entity when it is true, or a general entity type when it is false.

    OUTPUT FORMAT

    CRITICAL: Your response must be a valid JSON object with the following structure:

    {{
    "ENTITY_KEY": {{
        "type": ["type1", "type2"],
        "lexical": {{"property": "value"}},
        "semantic": ["description1", "description2"],
        "constant": true/false
    }}
    }}

    FORMAT REQUIREMENTS: "type" is an array of strings. "lexical" is an object (can be empty {{}}). "semantic" is an array of strings (can be empty []). "constant" is a boolean (true or false, not a string).

    EXAMPLES:
    {examples}

    CRITICAL REQUIREMENTS

    Your response must be ONLY the JSON object with no additional text. Ensure the JSON is valid and properly formatted. Do NOT leave any fields incomplete or empty when they should contain values. Follow the exact data structure shown in examples. Check that arrays use [], objects use {{}}, and booleans are true/false (not strings).
    FOLLOW THE FORMAT EXACTLY.

    YOUR TASK

    Extract entities from the following query:

    Q: {query} """
    
    # 2. Convert the list of entity types into a comma-separated string for insertion.
    # e.g., ['disease', 'gene'] -> "disease, gene"
    if  dataset_name == "prime":
        entity_types_list = ["disease", "gene/protein", "molecular_function", "drug", "pathway", "anatomy", "effect/phenotype", "biological_process", "cellular_component", "exposure"]
    elif dataset_name == "mag":
        entity_types_list = ["paper", "author", "institution", "field_of_study"]
    elif dataset_name == "amazon":
        entity_types_list = ["product", "brand", "color","category"]
    else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

    with open(f"custom_pipeline/prompt_examples/entity_{dataset_name}.txt", "r") as file:
        example_string = file.read() 
    
    # 3. Format the base prompt by replacing the placeholder with the generated string.
    entity_types_string = ", ".join(entity_types_list)
    formatted_prompt = base_prompt.format(entity_types_placeholder=entity_types_string,examples=example_string,query=query)
    
    return formatted_prompt

def get_relation_extraction_prompt(dataset_name: str , natural_language_query: str, identified_entities_string: str) -> str:
    
    base_prompt = """RELATION EXTRACTION TASK
        OBJECTIVE: Given a natural language query Q and a set of identified entities from that query, identify all possible relations (edges) between these entities that are implied by the query.

        TASK REQUIREMENTS

        1. Identify all possible edges between entities that can be understood from the query. Consider all entity pairs.

        2. For each identified relation, provide a list of possible edge types. Since edges are undirected, if you think A-rel1->B and A<-rel2-B, include both relations as (A,B): [rel1, rel2] in your answer.

        3. The ANSWER entity must be associated with at least one of the other identified entities.

        EDGE TYPE IDENTIFICATION PROCESS

        Step 1: Consider Initial Edge Types
        Start by considering all semantically relevant edge types from this list: {relation_types_placeholder} THE RELATION TYPE SHOULD NOT BE ANY OTHER THAN THESE.
        DO NOT consider any other name for edge as edge type, other than the ones mentioned in the list above. if some other type of connection, say 'abc' is mentioned in the query between two entities, assign all the labels from the above list that are semantically close to 'abc'.

        Step 2: Filter Based on Query Semantics
        Only remove edge types that are semantically irrelevant or cannot be implied by the given query in any reasonable way.

        Step 3: Validate Against Node Types
        Consider the node types of the two entities between which the edge exists. Strictly follow the valid triplet patterns listed below.


        IMPORTANT: Only use edge types not listed in the triplets above if the query implies a relation that absolutely cannot be represented using the standard triplets.

        VALID TRIPLET PATTERNS
        {valid_triplet_list}

        EXAMPLES
        {examples}

        OUTPUT FORMAT

        Your response must be a valid JSON object with the following structure:

        {{
        ("ENTITY_KEY_1", "ENTITY_KEY_2"): ["edge_type_1", "edge_type_2"],
        ("ENTITY_KEY_3", "ENTITY_KEY_4"): ["edge_type_3"]
        }}

        Where ENTITY_KEY_1, ENTITY_KEY_2, etc. are the entity identifiers (A, B, C, ANSWER, etc.) and the edge types are strings from the valid triplet list.


        YOUR TASK

        Given the query Q and identified entities below, identify all relations possible between entities and provide them in the JSON format specified above. Do not give any other text in the output, just the required JSON formatted string.FOLLOW THE FORMAT EXACTLY
        Q :{natural_language_query} 
        Identified Entities: {identified_entities_string}
        """
     
    relation_types = [] 
    if dataset_name == "prime":
        relation_types = ['ppi', 'carrier', 'enzyme', 'target', 'transporter', 'contraindication', 'indication', 'off-label use', 'synergistic interaction', 'associated with', 'parent-child', 'phenotype absent', 'phenotype present', 'side effect', 'interacts with', 'linked to', 'expression present', 'expression absent']
    elif dataset_name == "mag":
        relation_types = ['writes', 'affiliated_with', 'cites', 'has_topic']
    elif dataset_name == "amazon":
        relation_types = ['also_buy', 'also_view', 'has_brand', 'has_category', 'has_color']
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    
    with open(f"custom_pipeline/prompt_examples/relation_{dataset_name}.txt", "r") as file:
        example_string = file.read() 
    
    with open(f"custom_pipeline/prompt_examples/{dataset_name}_valid_triplet_list.txt", "r") as file:
        valid_triplet_list = file.read()
     
    relation_types_string = ", ".join(relation_types)
    formatted_prompt = base_prompt.format(relation_types_placeholder=relation_types_string,
                                            valid_triplet_list=valid_triplet_list,
                                            examples=example_string,
                                            natural_language_query=natural_language_query,
                                            identified_entities_string=identified_entities_string)
        
    return formatted_prompt


# res = get_relation_extraction_prompt("mag","AAAAAAAA" , "BBBBBBBBBBB")

# print(res)