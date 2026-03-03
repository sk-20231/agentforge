# Simple Agent – Flow

## High-level flow (Mermaid)

```mermaid
flowchart TB
    subgraph cli [CLI]
        A[User input]
        A --> B[log_event: user_input]
        B --> C[run_agent]
        C --> Z[log_event: final_answer]
        Z --> Out[Print result]
    end

    subgraph run_agent [run_agent]
        C --> D[classify_intent]
        D --> E[log_event: intent_classification]
        E --> F{Intent?}
        F -->|REMEMBER| G[store_memory if candidate]
        G --> R1["Return: Got it 👍"]
        F -->|ACT| H[run_llm_with_tools]
        H --> I[Parse JSON reply]
        I --> J{store_memory?}
        J -->|yes| K[store_memory]
        J -->|no| R2[Return reply]
        K --> R2
        F -->|REACT| L[run_react_agent]
        L --> M[react_loop]
        M --> R3[Return reply]
        F -->|ANSWER or RESPOND_WITH_MEMORY| N[answer_with_memory]
        N --> R4[Return reply]
        F -->|IGNORE| R5["Return: Okay 🙂"]
    end
```

## Intent classification

```mermaid
flowchart LR
    In[user_input] --> LLM1[LLM: intent prompt]
    LLM1 --> Parse[json.loads]
    Parse --> Out["intent, memory_candidate, reason"]
```

## ACT path (single tool use)

```mermaid
flowchart TB
    A[run_llm_with_tools] --> B[get_relevant_memories]
    B --> C[build_messages: system + memories + user + OUTPUT_SCHEMA]
    C --> D[LLM with TOOLS_SCHEMA]
    D --> E{Tool calls?}
    E -->|No| F[Return message.content]
    E -->|Yes| G[execute_tool for each]
    G --> H[Follow-up LLM with tool results]
    H --> F
```

## REACT path (multi-step loop)

```mermaid
flowchart TB
    A[react_loop] --> B[get_relevant_memories]
    B --> C[build_prompt: user_input + memory_chunks]
    C --> D[Loop max_steps]
    D --> E[LLM: structured JSON]
    E --> F[Parse: thought, action, plan]
    F --> G{action.type?}
    G -->|final| H{store_memory?}
    H -->|yes| I[store_memory]
    H -->|no| J[Return reply]
    I --> J
    G -->|tool| K[execute_tool]
    K --> L[Append assistant + tool to messages]
    L --> D
    D -->|steps exhausted| M["Return: too many steps"]
```

## ANSWER / RESPOND_WITH_MEMORY path

```mermaid
flowchart LR
    A[answer_with_memory] --> B[load_memory]
    B --> C[Build messages: system + memory_text + user]
    C --> D[LLM]
    D --> E[Return message.content]
```

## Data flow summary

| Path      | Memory read              | Memory write        | Tools        |
|-----------|--------------------------|---------------------|-------------|
| REMEMBER  | —                        | store_memory        | —           |
| ACT       | get_relevant_memories    | optional store      | calculator  |
| REACT     | get_relevant_memories    | optional store      | calculator  |
| ANSWER    | load_memory (full)       | —                   | —           |
| IGNORE    | —                        | —                   | —           |
