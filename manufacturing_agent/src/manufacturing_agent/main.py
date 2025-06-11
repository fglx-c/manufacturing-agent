#!/usr/bin/env python
import sys
import warnings
import os

from manufacturing_agent.crew import ManufacturingAgentCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    # Try to get layer number from environment variable first, then prompt user
    layer_number = os.environ.get('LAYER_NUMBER')
    
    if not layer_number:
        layer_number = input("Enter the layer number (default: 1): ").strip()
        if not layer_number:
            layer_number = "1"
    
    print(f"Running manufacturing agent for layer {layer_number}")
    
    inputs = {
        'layer_number': layer_number,
        'control_options_path': '/scratch/multi-user/manufacturing-agent/manufacturing_agent/src/data',
        'output_path': '/scratch/multi-user/manufacturing-agent/manufacturing_agent/src/output'
    }
    
    try:
        ManufacturingAgentCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")