import json

from datasets import load_dataset

dataset = load_dataset("THUDM/AgentInstruct")["mind2web"]
df = dataset.to_pandas()

# Standardize the role names
ROLE_MAP = {
    # user
    "user": "user",
    "human": "user",
    # assistant
    "chatgpt": "assistant",
    "gpt": "assistant",
    "bard": "assistant",
    # system
    "system": "system",
}


def normalize_conv_turn(conv_turn):
    role = conv_turn["from"]
    content = conv_turn["value"]

    if role in ROLE_MAP:
        role = ROLE_MAP[role]
    else:
        raise NotImplementedError(f"Unrecognized role: {role}")

    return {"role": role, "content": content}


def normalize_conv(conv):
    res_conv = []
    for turn in conv:
        res_conv.append(normalize_conv_turn(turn))
    return res_conv


# print(
#     "Role count (once per turn):",
#     df["conversations"]
#     .apply(lambda x: set([turn["from"] for turn in x]))
#     .apply(lambda x: Counter(x)).sum()
# )

df["conversations"] = df["conversations"].apply(normalize_conv)

# print(f"Dataset size before filtering: {len(df)}")
df = df[df["conversations"].apply(lambda x: any(turn["role"] == "user" for turn in x))]
# print(f"Dataset size after filtering (has at least 1 user turn): {len(df)}")
df = df[df["conversations"].apply(lambda x: any(turn["role"] == "assistant" for turn in x))]
# print(f"Dataset size after filtering (has at least 1 assistant turn): {len(df)}")
df = df[df["conversations"].apply(lambda x: x[0]["role"] != "assistant")]
# print(f"Dataset size after filtering (first turn is not assistant): {len(df)}")

# find if there are consecutive assistant turns
df = df[
    df["conversations"].apply(
        lambda x: not any(
            x[i]["role"] == "assistant" and x[i + 1]["role"] == "assistant"
            for i in range(len(x) - 1)
        )
    )
]
# print(f"Dataset size after filtering (no conv with consecutive assistant turns): {len(df)}")
# print(
#     "Role count (once per turn, after normalization & filtering):",
#     df["conversations"]
#     .apply(lambda x: set([turn["role"] for turn in x]))
#     .apply(lambda x: Counter(x)).sum()
# )

df = df[["id", "conversations"]]
for index, row in df.iterrows():
    print(json.dumps(row.to_dict()))
