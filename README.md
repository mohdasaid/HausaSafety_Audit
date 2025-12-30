# Safe in the Future, Dangerous in the Past: Dissecting Temporal and Linguistic Vulnerabilities in LLMs

**Authors:** Muhammad Abdullahi Said¹, Muhammad Sammani Sani²  
¹*African Institute for Mathematical Science*, ²*University of Vienna*


## 📌 Abstract
As Large Language Models (LLMs) integrate into critical global infrastructure, the assumption that safety alignment transfers zero-shot from English to other languages remains a dangerous blind spot. This study presents a systematic audit of three state-of-the-art models (**GPT-5.1**, **Gemini 3 Pro**, and **Claude 4.5 Opus**) using **HausaSafety**, a novel adversarial dataset grounded in West African threat scenarios (e.g., "Yahoo-Yahoo" fraud, "Dane gun" manufacturing).

We identify a mechanism of **Complex Interference** where safety is determined by the intersection of language and temporal framing. Notably, we report a profound **Temporal Asymmetry**, where past-tense framing bypassed defenses significantly more often than future or present-tense prompts.

