# Dolci Instruct SFT Tool Use

## Description

Dolci-Instruct-SFT-Tool-Use is a tool-use instruction tuning dataset released by Allen AI for the Olmo 3 Instruct models. It contains multi-turn conversations where a user asks for help, the assistant emits one or more function calls, the environment returns tool results, and the assistant summarizes the answer.

The dataset focuses on:
- Function-calling trajectories with explicit tool signatures
- Single-turn and multi-turn tool-use conversations
- Parallel or batched tool calls followed by environment results
- Synthetic tool-use data used in the Olmo 3 instruction tuning mixture

## Dataset Information

- Source URL: https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use
- Main dataset card: https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT
- License: ODC-BY
- Split used: `train`
- Size: 227,579 examples in the Hugging Face dataset card

## Schema Mapping

- Raw `user` messages become `TextObservation` entries with `source="user"`.
- Raw assistant `function_calls` strings are parsed as Python-style calls and become `ApiAction` entries. Dotted tool names are converted to valid Python identifiers by replacing non-identifier characters with underscores.
- Raw `environment` messages become `TextObservation` entries with `source="environment"`; newline-separated tool results are split and interleaved with matching batched tool calls when counts align.
- Raw assistant natural-language responses become `MessageAction` entries. The final response is wrapped in `<finish>` tags for the OpenHands SFT converter.
- Per-example tool schemas from the raw system message are converted into Python stub functions and stored in `details.available_apis` so agent converters can expose the appropriate tool documentation.

## Known Limitations

- The converter parses function calls that are valid Python-style call expressions. Malformed calls are skipped with an error message rather than emitted as partial trajectories.
- Function and parameter names are normalized to Python identifiers for compatibility with ADP `ApiAction` and OpenHands tool formatting.
- Tool schema parameters are emitted as optional in generated stubs because the raw schemas often omit explicit `required` lists or encode defaults inconsistently.

## Citation

```bibtex
@misc{olmo2025olmo3,
title={Olmo 3},
author={Team Olmo and Allyson Ettinger and Amanda Bertsch and Bailey Kuehl and David Graham and David Heineman and Dirk Groeneveld and Faeze Brahman and Finbarr Timbers and Hamish Ivison and Jacob Morrison and Jake Poznanski and Kyle Lo and Luca Soldaini and Matt Jordan and Mayee Chen and Michael Noukhovitch and Nathan Lambert and Pete Walsh and Pradeep Dasigi and Robert Berry and Saumya Malik and Saurabh Shah and Scott Geng and Shane Arora and Shashank Gupta and Taira Anderson and Teng Xiao and Tyler Murray and Tyler Romero and Victoria Graf and Akari Asai and Akshita Bhagia and Alexander Wettig and Alisa Liu and Aman Rangapur and Chloe Anastasiades and Costa Huang and Dustin Schwenk and Harsh Trivedi and Ian Magnusson and Jaron Lochner and Jiacheng Liu and Lester James V. Miranda and Maarten Sap and Malia Morgan and Michael Schmitz and Michal Guerquin and Michael Wilson and Regan Huff and Ronan Le Bras and Rui Xin and Rulin Shao and Sam Skjonsberg and Shannon Zejiang Shen and Shuyue Stella Li and Tucker Wilde and Valentina Pyatkin and Will Merrill and Yapei Chang and Yuling Gu and Zhiyuan Zeng and Ashish Sabharwal and Luke Zettlemoyer and Pang Wei Koh and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
year={2025},
eprint={2512.13961},
archivePrefix={arXiv},
primaryClass={cs.CL},
url={https://arxiv.org/abs/2512.13961},
}
```
