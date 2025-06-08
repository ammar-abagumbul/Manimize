# Manimize

An AI-powered educational tool that automatically generates and executes Manim animations based on natural language prompts.

## Overview

This project leverages AI agents to create mathematical visualizations using Manim (Mathematical Animation Engine). Simply describe what you want to visualize, and the AI agent will generate the appropriate Manim code, execute it, and produce the animation.

## Features

- **Natural Language Input**: Describe mathematical concepts in plain English
- **Automated Code Generation**: AI generates valid Manim Python code
- **Automatic Execution**: Code is executed automatically to produce animations
- **Error Handling**: AI can debug and fix code issues automatically
- **Educational Focus**: Designed for teaching and learning mathematical concepts

## Example Usage

**Input**: "Show me the different types of triangles"

**Output**: The AI agent will:
1. Generate Manim code to visualize equilateral, isosceles, and scalene triangles
2. Write the code to a Python file
3. Execute the code to create the animation
4. Handle any errors that occur during execution

## How It Works

1. **User Input**: Provide a natural language description of what you want to visualize
2. **Code Generation**: AI agent creates appropriate Manim code
3. **File Writing**: Code is saved to a Python file
4. **Execution**: Manim renders the animation
5. **Error Handling**: If execution fails, the agent debugs and fixes issues

## Requirements

- Python 3.8+
- Manim Community Edition
- LangChain or LangGraph
- OpenAI API key or Azure AI access

## Use Cases

- **Mathematics Education**: Visualize geometric concepts, functions, transformations
- **Teaching Aids**: Create custom animations for specific lessons
- **Rapid Prototyping**: Quickly generate mathematical visualizations
- **Interactive Learning**: Students can request visualizations of concepts they're studying

## Limitations

- Requires stable internet connection for AI model access
- Generated code quality depends on AI model capabilities
- Complex mathematical concepts may require manual refinement
- Currently focused on 2D mathematical visualizations

## License

MIT License - Feel free to use for educational purposes.

## Roadmap

- [ ] Web interface for easier access
- [ ] Integration with educational platforms
- [ ] Support for multiple animation styles
- [ ] Fine-tuning of AI model for better code generation

