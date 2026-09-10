# AI Agents

A collection of practical **AI agent projects** built with Python and modern agent frameworks. The projects explore RAG, multi-agent collaboration, task planning, debate, and software engineering workflows.

## Projects

```text
├── 1_modern_resume/
├── 2_software_team/
├── 3_debate_crew/
└── 4_engineering_crew/
```

### 1. Modern Resume

An AI-powered resume assistant that lets users ask questions about Sadegh's professional background.

It uses:

* RAG and a knowledge base
* Resume and project information
* Gradio for the chat interface
* Code extraction and project knowledge building
* OpenAI-based question answering

The application is launched through `app.py` and provides a conversational interface.

### 2. Software Team

A group of specialized agents that work together to turn a software idea into a development roadmap.

The agents focus on:

* Product definition
* Requirements
* Technical planning
* Roadmap creation

The project also includes a notebook for generating the roadmap.

### 3. Debate Crew

A **CrewAI** multi-agent system designed for research and debate.

Agents collaborate through configured roles and tasks to research a topic and produce a final report. The project includes agent configuration, tasks, tools, and the main crew logic.

### 4. Engineering Crew

A multi-agent software engineering workflow based on **CrewAI**.

The project uses specialized agents and tasks to simulate an engineering team and solve more complex software development problems. Configuration, tools, crew logic, and execution are separated into dedicated modules.

## Technologies

* Python
* OpenAI APIs
* CrewAI
* Gradio
* RAG
* Jupyter Notebook
* UV

## Getting Started

Clone the repository:

```bash
git clone https://github.com/SadeghMahmoudAbadi/Agents.git
cd Agents
```

Each project has its own dependencies and setup instructions. For the CrewAI projects, install the dependencies with `uv` and configure the required API keys before running the agents.

## Purpose

This repository is mainly a practical collection for learning and experimenting with:

* AI agents
* Multi-agent systems
* RAG
* LLM workflows
* Agent collaboration
* AI-assisted software development
